from __future__ import annotations

import base64
import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from flask import Flask, jsonify, render_template, request, session

from .agent import AgentReply, ConversationState, SupportAgent
from .config import Settings
from .voice import GnaniVoice, SUPPORTED_AUDIO, language_code_for_reply


LOGGER = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 25 * 1024 * 1024


@dataclass
class WebConversation:
    agent: SupportAgent
    state: ConversationState = field(default_factory=ConversationState)
    voice: GnaniVoice | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_app(
    settings: Settings,
    agent_factory: Callable[[Settings], SupportAgent] = SupportAgent,
    voice_factory: Callable[[Settings], GnaniVoice] = GnaniVoice,
) -> Flask:
    template_dir = Path(__file__).with_name("web_templates")
    static_dir = Path(__file__).with_name("web_static")
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
        static_url_path="/static",
    )
    app.secret_key = uuid.uuid4().hex
    app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES

    conversations: dict[str, WebConversation] = {}
    conversations_lock = threading.Lock()

    def get_conversation() -> WebConversation:
        conversation_id = session.get("conversation_id")
        if not isinstance(conversation_id, str):
            conversation_id = uuid.uuid4().hex
            session["conversation_id"] = conversation_id
        with conversations_lock:
            conversation = conversations.get(conversation_id)
            if conversation is None:
                conversation = WebConversation(agent=agent_factory(settings))
                conversations[conversation_id] = conversation
        return conversation

    def reply_payload(reply: AgentReply) -> dict[str, object]:
        return {
            "reply": reply.text,
            "decision": reply.decision,
            "evidence": list(reply.evidence),
        }

    def failure_message(exc: Exception, *, voice: bool = False) -> str:
        status_code = getattr(exc, "status_code", None)
        error_name = type(exc).__name__
        service = "voice service" if voice else "AI service"
        if status_code == 429 or error_name == "RateLimitError":
            return f"The {service} is rate-limited right now. Wait a little and try again."
        if status_code == 400:
            return f"The {service} couldn’t format a reply. Please send your question again."
        if status_code is not None and status_code >= 500 or error_name == "APIConnectionError":
            return f"The {service} is temporarily unavailable. Please try again shortly."
        return "I couldn’t complete that request. Please try again."

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({
            "llm_configured": bool(settings.openai_api_key),
            "voice_configured": bool(settings.gnani_api_key),
        })

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object containing a message."}), 400
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Type a message before sending."}), 400
        if len(message) > 4000:
            return jsonify({"error": "Messages must be 4,000 characters or fewer."}), 400

        conversation = get_conversation()
        try:
            with conversation.lock:
                reply, conversation.state = conversation.agent.respond(message.strip(), conversation.state)
        except Exception as exc:
            LOGGER.exception("Web chat turn failed.")
            return jsonify({"error": failure_message(exc)}), 502
        return jsonify(reply_payload(reply))

    @app.post("/api/voice")
    def voice():
        audio_file = request.files.get("audio")
        if audio_file is None or not audio_file.filename:
            return jsonify({"error": "Choose an audio file to send."}), 400
        suffix = Path(audio_file.filename).suffix.casefold()
        if suffix not in SUPPORTED_AUDIO:
            return jsonify({"error": "Use a WAV, MP3, M4A, FLAC, or OGG audio file."}), 400
        audio_bytes = audio_file.read(MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            return jsonify({"error": "The selected audio file is empty."}), 400
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return jsonify({"error": "Audio files must be 25 MB or smaller."}), 413

        conversation = get_conversation()
        try:
            with conversation.lock:
                if conversation.voice is None:
                    conversation.voice = voice_factory(settings)
                with tempfile.TemporaryDirectory(prefix="voiceops_web_") as temp_dir:
                    audio_path = Path(temp_dir) / f"customer{suffix}"
                    audio_path.write_bytes(audio_bytes)
                    transcript = conversation.voice.transcribe(audio_path)
                    if transcript:
                        reply, conversation.state = conversation.agent.respond(
                            transcript,
                            conversation.state,
                        )
                    else:
                        reply = AgentReply(
                            "Sorry, I didn’t catch that. Please try another recording.",
                            "clarify",
                        )
                    output_path = Path(temp_dir) / "reply.wav"
                    conversation.voice.synthesize(
                        reply.text,
                        output_path,
                        language_code_for_reply(transcript) if transcript else settings.gnani_language_code,
                    )
                    payload = reply_payload(reply)
                    payload.update({
                        "transcript": transcript,
                        "audio_base64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
                    })
        except Exception as exc:
            LOGGER.exception("Web voice turn failed.")
            return jsonify({"error": failure_message(exc, voice=True)}), 502
        return jsonify(payload)

    @app.post("/api/reset")
    def reset():
        conversation_id = session.get("conversation_id")
        if isinstance(conversation_id, str):
            with conversations_lock:
                conversations.pop(conversation_id, None)
        session.pop("conversation_id", None)
        return jsonify({"ok": True})

    return app


def run_web(settings: Settings, port: int = 5000) -> None:
    app = create_app(settings)
    LOGGER.info("Starting local web interface at http://127.0.0.1:%s", port)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

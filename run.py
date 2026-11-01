from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.agent import ConversationState, SupportAgent
from src.config import Settings
from src.evaluate import run_evaluation
from src.report import create_report
from src.voice import GnaniVoice, language_code_for_reply
from src.web import run_web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("voiceops")


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return stem or "audio"


def batch_audio(settings: Settings, input_dir: Path, output_dir: Path) -> int:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for voice conversations. Configure it in .env.")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input audio folder does not exist: {input_dir}")
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.casefold() in {
        ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    })
    if not files:
        raise FileNotFoundError(f"No supported audio files were found in {input_dir}")
    voice = GnaniVoice(settings)
    agent = SupportAgent(settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = 0
    for audio_path in files:
        state = ConversationState()
        transcript = voice.transcribe(audio_path)
        if transcript:
            reply, state = agent.respond(transcript, state)
            reply_text = reply.text
            decision = reply.decision
            evidence = list(reply.evidence)
            tool_calls = list(reply.tool_calls)
        else:
            reply_text = "Sorry, I didn’t catch that. Could you say it again?"
            decision = "clarify"
            evidence = []
            tool_calls = []
        stem = _safe_stem(audio_path.name)
        audio_output = output_dir / f"{stem}_reply.wav"
        voice.synthesize(
            reply_text,
            audio_output,
            language_code_for_reply(transcript) if transcript else settings.gnani_language_code,
        )
        record = {
            "input_audio": audio_path.name,
            "transcript": transcript,
            "reply": reply_text,
            "decision": decision,
            "evidence": evidence,
            "tool_calls": tool_calls,
            "reply_audio": audio_output.name if transcript else None,
        }
        (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / f"{stem}_transcript.txt").write_text(transcript, encoding="utf-8")
        (output_dir / f"{stem}_reply.txt").write_text(reply_text, encoding="utf-8")
        successful += 1
        LOGGER.info("Processed %s; decision=%s; sources=%s", audio_path.name, decision, ",".join(evidence))
    print(f"Processed {successful} audio file(s). Outputs: {output_dir.resolve()}")
    return 0


def chat(settings: Settings) -> int:
    agent = SupportAgent(settings)
    state = ConversationState()
    print("ZentroMart AI support. Type 'quit' to end.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.casefold() in {"quit", "exit"}:
            return 0
        try:
            reply, state = agent.respond(message, state)
        except Exception as exc:
            LOGGER.error("The turn failed: %s", exc)
            print(f"Agent error: {exc}")
            continue
        print(f"Agent: {reply.text}")
        if reply.evidence:
            print("Evidence:", ", ".join(reply.evidence))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps support agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    batch = subparsers.add_parser("batch", help="Transcribe, answer, and synthesize every audio file in a folder")
    batch.add_argument("--input", type=Path, required=True, help="Folder containing customer audio")
    batch.add_argument("--output", type=Path, default=Path("outputs"), help="Folder for transcripts, replies, sources, and WAVs")
    subparsers.add_parser("chat", help="Run a text support conversation in the terminal")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate supplied practice questions and conversations")
    evaluate.add_argument("--output", type=Path, default=Path("sample_outputs"), help="Where evaluation JSON is saved")
    evaluate.add_argument("--with-llm", action="store_true", help="Reserved for credentialed live answer scoring")
    report = subparsers.add_parser("report", help="Generate the four-page capstone report PDF")
    report.add_argument("--evaluation", type=Path, default=Path("sample_outputs/practice_evaluation.json"))
    report.add_argument("--output", type=Path, default=Path("report.pdf"))
    web = subparsers.add_parser("web", help="Run the local browser-based support interface")
    web.add_argument("--port", type=int, default=5000, help="Local web server port")
    args = parser.parse_args(argv)
    settings = Settings()

    if args.command == "batch":
        return batch_audio(settings, args.input, args.output)
    if args.command == "chat":
        return chat(settings)
    if args.command == "evaluate":
        result = run_evaluation(settings, args.output, with_llm=args.with_llm)
        print(json.dumps({key: value for key, value in result.items() if key not in {"questions", "conversations"}}, indent=2))
        print(f"Detailed expected/actual results saved to {args.output / 'practice_evaluation.json'}")
        return 0
    if args.command == "report":
        if not args.evaluation.is_file():
            parser.error(f"Evaluation JSON not found: {args.evaluation}. Run `python run.py evaluate` first.")
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
        create_report(evaluation, args.output)
        print(f"Report generated: {args.output.resolve()}")
        return 0
    if args.command == "web":
        run_web(settings, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

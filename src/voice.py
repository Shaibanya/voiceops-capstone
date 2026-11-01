from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

from .config import Settings


LANGUAGE_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "hi-en": "hi-en",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class GnaniVoice:
    """Thin adapter around the official gnani-vachana Python SDK."""

    def __init__(self, settings: Settings):
        if not settings.gnani_api_key:
            raise RuntimeError(
                "GNANI_API_KEY is not configured. Add your Gnani key to .env before running voice mode."
            )
        os.environ["GNANI_API_KEY"] = settings.gnani_api_key
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise RuntimeError(
                "The official Gnani SDK is missing. Install it with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        self.stt: Any = GnaniSTTClient()
        self.tts: Any = GnaniTTSClient()
        self.settings = settings

    @staticmethod
    def _validate_wav_duration(audio_path: Path) -> None:
        if audio_path.suffix.casefold() != ".wav":
            return
        try:
            with wave.open(str(audio_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (wave.Error, OSError) as exc:
            raise ValueError(f"Could not read WAV audio file {audio_path.name}: {exc}") from exc
        if duration > 60:
            raise ValueError(f"{audio_path.name} is {duration:.1f}s; Gnani STT accepts clips up to 60 seconds.")

    def transcribe(self, audio_path: Path, language_code: str | None = None) -> str:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        self._validate_wav_duration(audio_path)
        result = self.stt.transcribe(
            str(audio_path),
            language_code=language_code or self.settings.gnani_language_code,
        )
        transcript = result.get("transcript", "") if isinstance(result, dict) else ""
        if not isinstance(transcript, str):
            raise RuntimeError(f"Gnani STT returned an unexpected response for {audio_path.name}.")
        return transcript.strip()

    def synthesize(self, text: str, output_file: Path, language_code: str) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.tts.synthesize(
            text,
            voice=self.settings.gnani_tts_voice,
            model=self.settings.gnani_tts_model,
            language=language_code,
            output_file=str(output_file),
        )
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise RuntimeError(f"Gnani TTS did not create an audio file at {output_file}.")


def language_code_for_reply(text: str) -> str:
    from .agent import SupportAgent

    language = SupportAgent._language(text)
    if language == "or":
        return "or-IN"
    return LANGUAGE_CODES.get(language, "en-IN")

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def find_dataset_dir() -> Path:
    configured = os.getenv("DATASET_DIR")
    candidates = (
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026",
        PROJECT_ROOT.parent / "Capstone_Project_Dataset_FTTT_2026",
        PROJECT_ROOT,
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "kb").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()
    expected = PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026"
    raise FileNotFoundError(
        "Could not find the supplied dataset. Extract it beside the project, "
        "or set DATASET_DIR to its folder. Expected: " + str(expected)
    )


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = find_dataset_dir()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
    gnani_api_key: str | None = os.getenv("GNANI_API_KEY") or None
    gnani_tts_model: str = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
    gnani_tts_voice: str = os.getenv("GNANI_TTS_VOICE", "Kaveri")
    gnani_language_code: str = os.getenv("GNANI_LANGUAGE_CODE", "en-IN")


def create_openai_client(settings: Settings) -> OpenAI:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to create the LLM client.")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

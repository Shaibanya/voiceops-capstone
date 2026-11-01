Markdown
# VoiceOps 2026 — ZentroMart Voice Support Agent

**GitHub Repository:** [https://github.com/Shaibanya/voiceops-capstone.git](https://github.com/Shaibanya/voiceops-capstone.git)

This project is a local browser-based support agent for the supplied **synthetic** ZentroMart capstone dataset. It uses section-aware retrieval over all 21 supplied knowledge documents, the supplied CSV catalog, and identity-gated mock order tools. It features a Groq-hosted, OpenAI-compatible LLM for natural-language replies, and uses **only Gnani's official Python SDK** for speech recognition and speech generation.

The project files live beside (not inside) the supplied dataset folder. The dataset is neither copied nor modified by setup. This workspace already has the folder at `Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026`. If you extract it elsewhere, set `DATASET_DIR` in `.env`.

## What is Implemented

*   **Section-Aware RAG:** RAG chunks retain section IDs, source, document type, status, and effective date. Retrieval follows the precedence documented in `KB-00.1`; superseded material is excluded except where a dated pre-August 2026 electronics-return question needs it.
*   **Gated Order Lookup:** Product facts are looked up from `products.csv`. Order facts come only from the supplied `lookup_order` tool and stay hidden until the caller provides a matching phone suffix or delivery pincode. The conversation allows one retry.
*   **Human Handoff:** The supplied `create_ticket` tool is used for supported handovers. A ticket is reported as created only when the tool returns `created=True`.
*   **State & Privacy:** Conversation state is held in memory for follow-ups. Verification digits are redacted before history or handover summaries are retained.
*   **Voice Processing:** Voice batch mode calls `GnaniSTTClient` and `GnaniTTSClient` from `gnani-vachana`. STT clips are limited to 60 seconds; TTS explicitly selects `timbre-v2.5` and `Kaveri` by default.
*   **Local Frontend:** The local Flask frontend supports text chat, evidence citations, conversation reset, and recorded-audio uploads with spoken Gnani replies. It binds to `127.0.0.1`; it does not provide live microphone streaming.
*   **Automated Evaluation:** Offline evaluation records expected/actual decisions, evidence, and tool-call names for all supplied questions and conversation turns (verification values are redacted). With API credentials, `--with-llm` also compares prompt-only (Step 1) and RAG (Step 2) generated answers against required/forbidden facts.

> **Note:** The local, offline answer-decision evaluator is a baseline—not an LLM quality score. Without completed live model requests, it does not fabricate generated answers or Step 1/2 answer scores. The PDF report states that distinction.

## Current Verification

*   The dataset validator passes; the latest offline run covers 52 questions and 96 conversation turns. Its scores are offline decision/retrieval baselines, not generated-answer quality scores.
*   All 27 automated tests pass.
*   The local browser page and text chat were exercised. A live Groq JSON check and an order-specific cancellation answer succeeded, including retry after a provider JSON-formatting error.
*   The browser provides push-to-talk microphone recording (up to 60 seconds) and audio-file upload. A real Gnani audio round trip still needs to be tested.
*   The full live Step 1/Step 2 evaluation did not finish due to provider rate limits followed by a connection error. The report therefore claims no live comparison scores.
*   Voice upload and Gnani STT/TTS are implemented, but an end-to-end audio upload has not yet been verified with a recording.

## Requirements

*   **Python:** 3.11–3.14 (Python 3.14.3 was present and checked in this workspace).
*   **Gnani API Key:** Required for voice processing.
*   **Groq API Key:** Required for natural-language replies and live generated-answer evaluation. The OpenAI-compatible base URL and model are configurable via `OPENAI_BASE_URL` and `OPENAI_MODEL`. Dataset validation and offline retrieval/decision evaluation do not need either key.
*   **Audio Privacy:** No audio recordings are included in the supplied dataset. Use only a recording of your own voice unless other speakers have given written consent.

*The inspected official Gnani SDK release `gnani-vachana==0.9.0` declares `Requires-Python: >=3.9`; its dependency metadata is compatible with the project's declared Python range. Install the project requirements in a virtual environment.*

## Setup Instructions (Windows PowerShell)

Open PowerShell in this project folder and run:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
In .env, add your own keys (never commit this file). OPENAI_API_KEY should contain your Groq API key; use a model available to your Groq account:

Plaintext
OPENAI_API_KEY=your_groq_api_key
OPENAI_BASE_URL=[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)
OPENAI_MODEL=qwen/qwen3.8-27b
GNANI_API_KEY=your_gnani_key
If PowerShell blocks activation, use the venv interpreter directly instead:

PowerShell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Note: The LLM client uses the configured base URL and model for both chat and live evaluation, so the two paths stay consistent. The Gnani key is read from GNANI_API_KEY; that one key is used for both Gnani speech services. For an extracted dataset in another location, uncomment DATASET_DIR in .env and set it to the absolute dataset directory.

Run the Application
Start the browser interface on your computer:

PowerShell
python run.py web
Open http://127.0.0.1:5000. Ask a text question, use a suggested question, or choose Record voice to speak into your microphone and then Stop & send. You can also choose Send a voice recording to upload a WAV, MP3, M4A, FLAC, or OGG file. Microphone recordings are limited to 60 seconds, converted to WAV in the browser, and returned as playable Gnani speech. This is push-to-talk (one recorded turn at a time), not continuous live streaming.Chat requires OPENAI_API_KEY; voice also requires GNANI_API_KEY. The local web server binds only to 127.0.0.1.

To use another port:

PowerShell
python run.py web --port 5051
Run a terminal text conversation:

PowerShell
python run.py chat
Process every supported audio file (.wav, .mp3, .m4a, .flac, .ogg) in a folder. For each file, this saves a transcript, reply text, JSON with the evidence and decisions, and a Gnani-generated reply WAV:

PowerShell
python run.py batch --input .\my_audio --output .\outputs
(The input folder must contain at least one audio file. STT uses en-IN by default; set GNANI_LANGUAGE_CODE to another supported Gnani language code in .env when appropriate).

Validate and Evaluate
Run the supplied validator without changing the dataset:

PowerShell
python .\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
Run offline practice evaluation (no API keys required):

PowerShell
python run.py evaluate --output .\sample_outputs
(Writes sample_outputs\practice_evaluation.json, with expected versus actual decisions, evidence IDs, and redacted expected/actual tool-call traces).

After configuring OPENAI_API_KEY, run all 52 single-turn questions with both a prompt-only baseline and RAG answers, plus the 96 conversation turns:

PowerShell
python run.py evaluate --with-llm --output .\sample_outputs
(Live evaluation makes many external LLM requests and may incur charges or encounter provider rate limits. If a provider error stops the run, do not treat the previous JSON or PDF as a fresh live score).

Generate or refresh the report after evaluation:

PowerShell
python run.py report --evaluation .\sample_outputs\practice_evaluation.json --output .\report.pdf
Run focused code tests:

PowerShell
python -m unittest discover -s tests -v
Deliverables and Submission
run.py, src\, requirements.txt, .env.example, README.md and tests\.

report.pdf is generated locally by the command above and is limited to four pages.

demo_guidance.md provides a short, reproducible demo outline. Record the demo yourself after configuring credentials.

sample_outputs\ holds the evaluation JSON and generated batch results.

Submission Rules: Do not include .env, .venv, API keys, customer recordings, or a duplicate of the supplied dataset in your final submission archive. The dataset tools use synthetic data and are for this capstone only. Their phone/pincode checks are not production authentication.
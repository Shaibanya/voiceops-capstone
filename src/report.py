from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_report(evaluation: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="VoiceOps 2026 ZentroMart Capstone Report",
        author="VoiceOps Capstone Project",
    )
    styles = getSampleStyleSheet()
    step_comparison = evaluation.get("step_1_vs_step_2", {})
    step1_score = step_comparison.get("step_1_prompt_only", "not measured")
    step2_score = step_comparison.get("step_2_rag", "not measured")
    live_evaluation = evaluation.get("llm_generation_tested") is True
    step_summary = (
        f"Live answer-quality checks completed for the supplied set. Step 1 "
        f"prompt-only passed {step1_score}; Step 2 RAG passed {step2_score}. "
        "These are automatic required/forbidden-fact checks, not a human quality rating."
        if live_evaluation else
        "Step 1 prompt-only and generated-answer Step 2 quality were not measured in "
        "this evaluation. The table reports offline retrieval and decision baselines "
        "only; no generated-answer score is inferred from them."
    )
    story = [
        Paragraph("VoiceOps 2026 — ZentroMart Support Agent", styles["Title"]),
        Paragraph("Project, run guide and evaluation report", styles["Heading2"]),
        Paragraph(
            "This project implements section-aware policy retrieval, verified order lookup, "
            "human-ticket handoff, a Groq-hosted OpenAI-compatible conversational layer, "
            "and Gnani speech processing. A local browser interface supports text chat, "
            "evidence citations, session reset, microphone push-to-talk, and audio uploads.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph("Evaluation snapshot", styles["Heading2"]),
    ]
    questions = evaluation["questions"]
    conversations = evaluation["conversations"]
    data = [
        ["Measure", "Observed"],
        ["Practice questions", str(questions["count"])],
        ["Question decision baseline", f'{questions["decision_matches"]}/{questions["count"]}'],
        [
            "Question gold-evidence coverage",
            f'{questions["evidence_matches"]}/{questions["expected_evidence_cases"]}',
        ],
        [
            "ASR-text evidence coverage",
            f'{questions["asr_evidence_matches"]}/{questions["asr_evidence_cases"]}',
        ],
        ["Conversation turns", str(conversations["turns"])],
        [
            "Conversation decision baseline",
            f'{conversations["decision_matches"]}/{conversations["turns"]}',
        ],
        [
            "Conversation expected tool-call names",
            f'{conversations["tool_call_name_matches"]}/{conversations["turns"]}',
        ],
    ]
    table = Table(data, colWidths=[3.5 * inch, 2.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14324b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d2dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        table,
        Spacer(1, 12),
        Paragraph("Step 1 vs Step 2", styles["Heading2"]),
        Paragraph(step_summary, styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Design and safeguards", styles["Heading2"]),
        Paragraph(
            "Knowledge chunks retain section IDs and document metadata. Current policy, agent guide, "
            "help, FAQ and marketing precedence follows KB-00.1; superseded policy is excluded from "
            "ordinary retrieval. Order details remain gated behind the supplied mock verification "
            "tool, with one retry. Caller verification digits are redacted from stored conversation "
            "history, evidence files and ticket summaries. Support logs are not used as policy.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("What went wrong and next steps", styles["Heading2"]),
        Paragraph(
            "A live Groq JSON-mode check and an order-aware browser chat turn succeeded, including "
            "recovery from one provider JSON-formatting error. The full live practice evaluation "
            "did not complete because of provider rate limits and a connection error, so no Step 1 "
            "vs Step 2 scores are claimed here. The Gnani microphone/audio round trip still needs a "
            "consented recording test. Run `python run.py web` and open "
            "`http://127.0.0.1:5000`; then test Gnani audio and retry the full evaluation in "
            "smaller, rate-limited batches. Offline baselines are heuristics, not a substitute for "
            "human/native-language quality review.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Run the project", styles["Heading2"]),
        Paragraph(
            "Install dependencies with `python -m pip install -r requirements.txt`; if `.env` "
            "does not already exist, copy `.env.example` to `.env` and add your own Groq and "
            "Gnani keys locally. Start the "
            "browser UI with `python run.py web`. For folder-based speech processing use "
            "`python run.py batch --input .\\\\my_audio --output .\\\\outputs`. See README.md "
            "for setup, evaluation, and report commands.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(f"Generated from the supplied synthetic dataset. Evaluation date: {evaluation['generated_at']}.", styles["Italic"]),
    ])
    document.build(story)

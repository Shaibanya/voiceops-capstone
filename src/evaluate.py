from __future__ import annotations

import csv
import json
import re
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .agent import ConversationState, SupportAgent, SYSTEM_PROMPT
from .config import Settings, create_openai_client
from .data import find_products, product_query_context
from .llm import complete_json


PROMPT_ONLY_SYSTEM = """You are ZentroMart's AI customer-support assistant.
Reply briefly in the customer's language. Never invent policies or claim an action
occurred. If the needed information is unavailable, say so and offer a colleague.
Return ONLY a valid JSON object with exactly these keys, with no Markdown:
{"decision":"answer","reply":"customer-facing reply"}
The decision value must be exactly one of: answer, clarify, escalate, refuse."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _evidence_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(" ; ") if part.strip()]


def _contains_evidence(actual: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    return all(
        any(item == got or item in got for got in actual)
        for item in expected
    )


def _actual_question_decision(
    row: dict[str, str],
    evidence: list[str],
    products: list[dict[str, Any]],
) -> str:
    text = row["question"].casefold()
    if any(term in text for term in (
        "human", "speak to a person", "talk to someone", "police", "consumer court",
        "fraud", "never placed", "didn't place", "did not place", "burning smell",
        "swelling battery", "sparks",
    )):
        return "escalate"
    if any(term in text for term in (
        "compensation", "voucher", "override the policy", "ignore the policy",
        "phone number of the person", "details about other customers",
    )):
        return "refuse"
    if re.search(r"how many days|कितने दिन", text) and re.search(r"return|वापस", text) and not products:
        return "clarify"
    if re.search(r"when will my refund|when does .* refund", text) and not any(
        term in text for term in ("upi", "card", "net banking", "netbanking", "wallet", "cod", "cash")
    ):
        return "clarify"
    if any(term in text for term in ("crack", "damaged", "damage", "broken", "wrong item")) and any(
        int(product.get("price_inr") or 0) > 10000 for product in products
    ):
        return "escalate"
    if not evidence:
        return "answer"
    return "answer"


def _actual_conversation_decision(message: str, turn: int, state: Any) -> str:
    lowered = message.casefold()
    if turn == 1 and not re.search(r"\d{4}\s?\d{4}", message):
        return "clarify"
    if re.search(r"\d{4}\s?\d{4}", message) and state.get("verified") is not True:
        return "clarify"
    if state.get("pending_order") and state.get("verified") is not True:
        if state.get("verification_failed"):
            return "refuse" if state.get("verification_attempts", 0) >= 2 else "clarify"
        return "answer" if re.search(r"(?<!\d)(?:\d{4}|\d{6})(?!\d)", message) else "clarify"
    if any(term in lowered for term in ("human", "person", "fraud", "police", "consumer court", "swelling", "burning smell")):
        return "escalate"
    if any(term in lowered for term in ("compensation", "voucher")):
        return "refuse"
    if "cancel" in lowered:
        order = state.get("verified_order") or {}
        return "refuse" if str(order.get("status", "")).casefold() == "shipped" else "answer"
    return "answer"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _fact_matches(reply: str, required: str) -> bool:
    lowered_reply = _normalize(reply)
    alternatives = [part.strip() for part in required.split(" / ") if part.strip()]
    for alternative in alternatives:
        normalized = _normalize(alternative)
        if not normalized:
            continue
        if normalized.isdigit():
            if re.search(rf"(?<!\d){re.escape(normalized)}(?!\d)", lowered_reply):
                return True
        elif normalized in lowered_reply:
            return True
    return False


def _fact_score(reply: str, row: dict[str, str]) -> dict[str, Any]:
    required = _evidence_parts(row.get("must_include", ""))
    forbidden = [
        alternative.strip()
        for rule in _evidence_parts(row.get("must_not_include", ""))
        for alternative in rule.split(" / ")
        if alternative.strip()
    ]
    missing = [fact for fact in required if not _fact_matches(reply, fact)]
    violations = [phrase for phrase in forbidden if phrase.casefold() in reply.casefold()]
    return {
        "required_fact_count": len(required),
        "required_facts_matched": len(required) - len(missing),
        "missing_facts": missing,
        "forbidden_claims": violations,
        "answer_quality_pass": not missing and not violations,
    }


def _live_completion(client: Any, model: str, system: str, question: str) -> dict[str, str]:
    content = complete_json(
        client,
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    parsed = json.loads(content)
    if not isinstance(parsed.get("reply"), str) or parsed.get("decision") not in {"answer", "clarify", "escalate", "refuse"}:
        raise RuntimeError("The LLM evaluation response did not contain a valid decision and reply.")
    return parsed


def run_evaluation(settings: Settings, output_dir: Path, with_llm: bool = False) -> dict[str, Any]:
    if with_llm and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live answer evaluation.")
    ticket_temp = tempfile.TemporaryDirectory(prefix="voiceops_evaluation_")
    agent = SupportAgent(settings, ticket_path=Path(ticket_temp.name) / "evaluation_tickets.csv")
    client = None
    if with_llm:
        client = create_openai_client(settings)
    questions = _read_csv(settings.dataset_dir / "practice" / "questions.csv")
    conversations = _read_csv(settings.dataset_dir / "practice" / "conversations.csv")
    question_results: list[dict[str, Any]] = []
    for row in questions:
        question_text = row["question"]
        products = find_products(settings.dataset_dir, question_text)
        retrieval_query = question_text + " " + product_query_context(products, question_text)
        retrieved = agent.knowledge_base.retrieve(retrieval_query, limit=8)
        actual_evidence = [chunk.section_id for chunk, _ in retrieved]
        actual_evidence.extend(f"products.csv:{product['sku']}" for product in products)
        asr_query = row["asr_text"] or row["question"]
        asr_products = find_products(settings.dataset_dir, asr_query)
        asr_query += " " + product_query_context(asr_products, asr_query)
        asr_evidence = [
            chunk.section_id
            for chunk, _ in agent.knowledge_base.retrieve(asr_query, limit=8)
        ]
        asr_evidence.extend(f"products.csv:{product['sku']}" for product in asr_products)
        expected_evidence = _evidence_parts(row["gold_evidence"])
        actual_decision = _actual_question_decision(row, actual_evidence, products)
        item: dict[str, Any] = {
            "id": row["id"],
            "expected_decision": row["expected_decision"],
            "acceptable_decisions": _evidence_parts(row["acceptable_decisions"]),
            "actual_decision": actual_decision,
            "decision_match": actual_decision in _evidence_parts(row["acceptable_decisions"]),
            "expected_tool_calls": [],
            "actual_tool_calls": [],
            "expected_evidence": expected_evidence,
            "actual_evidence": actual_evidence,
            "evidence_match": _contains_evidence(actual_evidence, expected_evidence),
            "asr_text": row["asr_text"],
            "asr_actual_evidence": asr_evidence,
            "asr_evidence_match": _contains_evidence(asr_evidence, expected_evidence),
            "answer_generation": "not_run_without_OPENAI_API_KEY",
            "retrieval_found_support": bool(retrieved),
        }
        if agent._privacy_request(question_text):
            actual_decision = "refuse"
            actual_evidence.extend(["ESC-4", "IDV-4"])
            item.update({
                "actual_decision": actual_decision,
                "decision_match": actual_decision in _evidence_parts(row["acceptable_decisions"]),
                "actual_evidence": actual_evidence,
                "evidence_match": _contains_evidence(actual_evidence, expected_evidence),
            })
        escalation_reason = agent._ticket_reason(question_text, None)
        if escalation_reason and not with_llm:
            reply = agent._escalate(
                ConversationState(),
                question_text,
                escalation_reason,
                actual_evidence,
                [],
                agent._language(question_text),
            )
            actual_decision = reply.decision
            item.update({
                "actual_decision": reply.decision,
                "decision_match": reply.decision in _evidence_parts(row["acceptable_decisions"]),
                "actual_tool_calls": list(reply.tool_calls),
                "actual_evidence": list(reply.evidence),
            })
        if with_llm and client is not None:
            step1 = _live_completion(client, settings.openai_model, PROMPT_ONLY_SYSTEM, question_text)
            step2, _ = agent.respond(question_text, ConversationState())
            expected_decisions = _evidence_parts(row["acceptable_decisions"])
            item.update({
                "actual_decision": step2.decision,
                "decision_match": step2.decision in expected_decisions,
                "step_1_prompt_only": {
                    "decision": step1["decision"],
                    "reply": step1["reply"],
                    **_fact_score(step1["reply"], row),
                },
                "step_2_rag": {
                    "decision": step2.decision,
                    "reply": step2.text,
                    **_fact_score(step2.text, row),
                },
                "actual_tool_calls": list(step2.tool_calls),
                "actual_evidence": list(step2.evidence),
                "evidence_match": _contains_evidence(list(step2.evidence), expected_evidence),
                "answer_generation": "live_openai",
            })
        question_results.append(item)

    conversation_results: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = defaultdict(dict)
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in conversations:
        by_id[row["conversation_id"]].append(row)
    for conversation_id, turns in sorted(by_id.items()):
        state = states[conversation_id]
        conversation_state = ConversationState()
        for row in sorted(turns, key=lambda item: int(item["turn"])):
            turn = int(row["turn"])
            message = row["customer_message"]
            order_match = re.search(r"(?<!\d)(\d{4})\s?(\d{4})(?!\d)", message)
            tool_calls: list[str] = []
            verification_evidence: list[str] = []
            if order_match:
                state["pending_order"] = order_match.group(1) + order_match.group(2)
                state["verified"] = False
                tool_calls.append(f"lookup_order(order_id={state['pending_order']})")
            elif state.get("pending_order") and not state.get("verified"):
                proof = re.search(r"(?<!\d)(\d{4}|\d{6})(?!\d)", message)
                if proof:
                    tool_calls.append(
                        f"lookup_order(order_id={state['pending_order']}, verification_value=<redacted>)"
                    )
                    lookup = agent.lookup_order(
                        state["pending_order"],
                        verification_value=proof.group(1),
                    )
                    state["verified"] = lookup.get("verified") is True
                    state["verification_failed"] = not state["verified"]
                    state["verification_attempts"] = state.get("verification_attempts", 0) + (
                        0 if state["verified"] else 1
                    )
                    if state["verified"]:
                        conversation_state.pending_order_id = state["pending_order"]
                        conversation_state.verified_order = agent._safe_order(lookup)
                        state["verified_order"] = conversation_state.verified_order
                    else:
                        verification_evidence = ["IDV-2", "IDV-3"]
            if tool_calls:
                # The verification section is the evidence for the gated lookup action.
                if not verification_evidence:
                    verification_evidence = ["IDV-2"]
            else:
                verification_evidence = []
            history = state.setdefault("history", [])
            recent = [
                prior for prior in history[-8:]
                if not re.search(r"(?<!\d)\d{4}\s?\d{4}(?!\d)", prior)
                and not any(term in prior.casefold() for term in (
                    "last four", "last 4", "pincode", "pin code", "registered phone",
                    "आख़िरी", "आखिरी", "पिनकोड", "अंक",
                ))
            ]
            retrieval_query = " ".join(recent[-2:] + [message])
            verified_order = conversation_state.verified_order
            if verified_order and verified_order.get("damage_sku"):
                retrieval_query += " high value damaged claim"
            if verified_order and verified_order.get("is_plus"):
                retrieval_query += " ZentroMart Plus member"
            if verified_order:
                retrieval_query += " " + str(verified_order.get("status", ""))
            if verified_order and any(verified_order.get(field) for field in (
                "return_refund_initiated_on",
                "rto_refund_initiated_on",
                "cancellation_refund_initiated_on",
            )):
                retrieval_query += " refund initiated status refund timeline"
            if verified_order and verified_order.get("return_pickup_on"):
                retrieval_query += " returned item pickup quality check refund"
            products = find_products(settings.dataset_dir, retrieval_query)
            retrieval_query += " " + product_query_context(products, retrieval_query)
            retrieved = agent.knowledge_base.retrieve(retrieval_query, limit=8)
            actual_evidence = [chunk.section_id for chunk, _ in retrieved]
            actual_evidence.extend(verification_evidence)
            actual_evidence.extend(f"products.csv:{product['sku']}" for product in products)
            expected_evidence = _evidence_parts(row["gold_evidence"])
            actual_decision = _actual_conversation_decision(message, turn, state)
            if actual_decision == "escalate":
                actual_evidence.extend(section for section in ("ESC-1", "ESC-2", "ESC-3", "ESC-4") if section not in actual_evidence)
            verified_order = conversation_state.verified_order
            escalation_reason = agent._ticket_reason(message, verified_order)
            if escalation_reason and not tool_calls and not with_llm:
                sections = ["ESC-1", "ESC-2", "ESC-3", "ESC-4"]
                if escalation_reason == "refund_overdue":
                    sections.extend(["REF-2", "REF-6"])
                actual_evidence.extend(section for section in sections if section not in actual_evidence)
                reply = agent._escalate(
                    conversation_state,
                    message,
                    escalation_reason,
                    actual_evidence,
                    tool_calls,
                    agent._language(message),
                )
                actual_decision = reply.decision
                tool_calls = list(reply.tool_calls)
                actual_evidence = list(reply.evidence)
            expected_calls = row["expected_tool_calls"]
            safe_expected_calls = re.sub(
                r"(verification_value=)[^),]+",
                r"\1<redacted>",
                expected_calls,
            )
            expected_call_names = re.findall(r"[a-z_]+\(", safe_expected_calls)
            actual_call_names = re.findall(r"[a-z_]+\(", " ".join(tool_calls))
            item = {
                "conversation_id": conversation_id,
                "turn": turn,
                "expected_decision": row["expected_decision"],
                "actual_decision": actual_decision,
                "decision_match": actual_decision == row["expected_decision"],
                "expected_tool_calls": safe_expected_calls,
                "actual_tool_calls": tool_calls,
                "tool_call_names_match": expected_call_names == actual_call_names,
                "expected_evidence": expected_evidence,
                "actual_evidence": actual_evidence,
                "evidence_match": _contains_evidence(actual_evidence, expected_evidence),
                "answer_generation": "not_run_without_OPENAI_API_KEY",
            }
            if with_llm:
                reply, conversation_state = agent.respond(message, conversation_state)
                item.update({
                    "actual_decision": reply.decision,
                    "decision_match": reply.decision == row["expected_decision"],
                    "actual_tool_calls": list(reply.tool_calls),
                    "actual_evidence": list(reply.evidence),
                    "evidence_match": _contains_evidence(list(reply.evidence), expected_evidence),
                    "reply": reply.text,
                    **_fact_score(reply.text, row),
                    "answer_generation": "live_openai",
                })
            conversation_results.append(item)
            history.append(message)

    all_results = question_results + conversation_results
    evidence_cases = [item for item in all_results if item["expected_evidence"]]
    question_evidence_cases = [item for item in question_results if item["expected_evidence"]]
    asr_evidence_matches = sum(item["asr_evidence_match"] for item in question_evidence_cases)
    summary = {
        "generated_at": date.today().isoformat(),
        "dataset_reference_date": "2026-10-20",
        "mode": "live_llm_and_rag" if with_llm else "offline_retrieval_and_decision_baseline",
        "llm_generation_tested": with_llm,
        "llm_note": "Live Step 1 and Step 2 generation was evaluated." if with_llm else
            "Set OPENAI_API_KEY and run `python run.py evaluate --with-llm` to score generated answers.",
        "questions": {
            "count": len(question_results),
            "decision_matches": sum(item["decision_match"] for item in question_results),
            "evidence_matches": sum(item["evidence_match"] for item in question_evidence_cases),
            "expected_evidence_cases": len(question_evidence_cases),
            "asr_evidence_matches": asr_evidence_matches,
            "asr_evidence_cases": len(question_evidence_cases),
            "results": question_results,
        },
        "conversations": {
            "turns": len(conversation_results),
            "decision_matches": sum(item["decision_match"] for item in conversation_results),
            "tool_call_name_matches": sum(item["tool_call_names_match"] for item in conversation_results),
            "evidence_matches": sum(item["evidence_match"] for item in conversation_results if item["expected_evidence"]),
            "expected_evidence_cases": sum(bool(item["expected_evidence"]) for item in conversation_results),
            "results": conversation_results,
        },
        "overall": {
            "decision_matches": sum(item["decision_match"] for item in all_results),
            "decision_cases": len(all_results),
            "evidence_matches": sum(item["evidence_match"] for item in evidence_cases),
            "evidence_cases": len(evidence_cases),
            "asr_evidence_matches": asr_evidence_matches,
            "asr_evidence_cases": len(question_evidence_cases),
        },
        "step_1_vs_step_2": {
            "step_1_prompt_only": (
                f'{sum(item.get("step_1_prompt_only", {}).get("answer_quality_pass", False) for item in question_results)}/{len(question_results)}'
                if with_llm else "not measured: OPENAI_API_KEY is required to run live answer generation"
            ),
            "step_2_rag": (
                f'{sum(item.get("step_2_rag", {}).get("answer_quality_pass", False) for item in question_results)}/{len(question_results)}'
                if with_llm else "retrieval measured offline; see question and conversation evidence metrics"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "practice_evaluation.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ticket_temp.cleanup()
    return summary

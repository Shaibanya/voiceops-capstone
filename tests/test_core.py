from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agent import AgentReply, ConversationState, SupportAgent, SYSTEM_PROMPT
from src.config import Settings, create_openai_client
from src.data import find_products, refund_is_overdue
from src.knowledge_base import KnowledgeBase
from src.evaluate import PROMPT_ONLY_SYSTEM, run_evaluation
from src.llm import complete_json
from src.voice import GnaniVoice
from src.web import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026"


def settings() -> Settings:
    return Settings(dataset_dir=DATASET, openai_api_key=None, gnani_api_key=None)


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase(DATASET / "kb")

    def test_refund_policy_outranks_conflicting_faq(self) -> None:
        results = self.kb.retrieve("How fast are refunds to the ZentroMart Wallet?")
        ids = [chunk.section_id for chunk, _ in results]
        self.assertIn("REF-2", ids)
        self.assertNotEqual(ids[0], "FAQ-1")

    def test_superseded_return_policy_is_excluded_for_current_question(self) -> None:
        ids = self.kb.sections_for("How many days to return electronics?")
        self.assertNotIn("RETv3-2", ids)
        self.assertIn("RET-2", ids)

    def test_historical_electronics_question_can_retrieve_superseded_policy(self) -> None:
        ids = self.kb.sections_for("What was the return window for electronics delivered 20 July?")
        self.assertIn("RET-3", ids)
        self.assertIn("RETv3-2", ids)

    def test_product_lookup_uses_supplied_catalog(self) -> None:
        matches = find_products(DATASET, "Is SKU ELE-0012 covered by ZentroMart Assured?")
        self.assertEqual([product["sku"] for product in matches], ["ELE-0012"])

    def test_product_lookup_ignores_catalog_color_suffix(self) -> None:
        matches = find_products(DATASET, "Can I return the Sootra Canvas Tote Bag?")
        self.assertIn("FSH-0038", [product["sku"] for product in matches])


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = SupportAgent(settings())

    def test_llm_prompt_shows_a_valid_decision_value(self) -> None:
        self.assertIn('{"decision":"answer","reply":"customer-facing reply"}', SYSTEM_PROMPT)
        self.assertIn("one of: answer, clarify, escalate, refuse", SYSTEM_PROMPT)
        self.assertNotIn('"answer|clarify|escalate|refuse"', SYSTEM_PROMPT)
        self.assertIn('{"decision":"answer","reply":"customer-facing reply"}', PROMPT_ONLY_SYSTEM)
        self.assertIn("one of: answer, clarify, escalate, refuse", PROMPT_ONLY_SYSTEM)

    def test_llm_client_uses_configured_provider_url(self) -> None:
        configured = Settings(
            dataset_dir=DATASET,
            openai_api_key="test-key",
            openai_base_url="https://api.groq.com/openai/v1",
        )
        with patch("openai.OpenAI") as client_factory:
            client = create_openai_client(configured)

        self.assertIs(client, client_factory.return_value)
        client_factory.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )

    def test_json_generation_failure_retries_without_provider_json_mode(self) -> None:
        class FakeBadRequestError(Exception):
            pass

        messages = [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Test"},
        ]
        client = Mock()
        retry_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"decision":"answer","reply":"Done."}')
                )
            ]
        )
        client.chat.completions.create.side_effect = [
            FakeBadRequestError("Failed to generate JSON."),
            retry_response,
        ]

        with patch("src.llm.BadRequestError", FakeBadRequestError):
            result = complete_json(client, "test-model", messages)

        self.assertEqual(result, '{"decision":"answer","reply":"Done."}')
        self.assertEqual(client.chat.completions.create.call_count, 2)
        first_call, retry_call = client.chat.completions.create.call_args_list
        self.assertEqual(first_call.kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", retry_call.kwargs)
        self.assertIn("syntactically valid JSON", retry_call.kwargs["messages"][0]["content"])
        self.assertEqual(messages[0]["content"], "Return JSON.")

    def test_non_json_fallback_is_repaired_before_returning(self) -> None:
        class FakeBadRequestError(Exception):
            pass

        messages = [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "How can I cancel my order?"},
        ]
        client = Mock()
        def response(content: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        client.chat.completions.create.side_effect = [
            FakeBadRequestError("Failed to generate JSON."),
            response("I can help with that. What is your order number?"),
            response('{"decision":"clarify","reply":"What is your order number?"}'),
        ]

        with patch("src.llm.BadRequestError", FakeBadRequestError):
            result = complete_json(client, "test-model", messages)

        self.assertEqual(
            result,
            '{"decision":"clarify","reply":"What is your order number?"}',
        )
        self.assertEqual(client.chat.completions.create.call_count, 3)
        repair_call = client.chat.completions.create.call_args_list[2]
        self.assertEqual(repair_call.kwargs["temperature"], 0)
        self.assertEqual(repair_call.kwargs["messages"][-2]["content"],
                         "I can help with that. What is your order number?")
        self.assertIn("valid JSON", repair_call.kwargs["messages"][-1]["content"])

    def test_non_json_repair_response_fails_explicitly(self) -> None:
        class FakeBadRequestError(Exception):
            pass

        client = Mock()
        client.chat.completions.create.side_effect = [
            FakeBadRequestError("Failed to generate JSON."),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain text"))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="still plain"))]),
        ]

        with patch("src.llm.BadRequestError", FakeBadRequestError):
            with self.assertRaisesRegex(RuntimeError, "not valid JSON after repair"):
                complete_json(
                    client,
                    "test-model",
                    [{"role": "system", "content": "Return JSON."}],
                )

    def test_order_lookup_requires_verification_and_redacts_proof(self) -> None:
        with (DATASET / "data" / "orders.csv").open(encoding="utf-8-sig", newline="") as handle:
            order = next(csv.DictReader(handle))
        with (DATASET / "data" / "customers.csv").open(encoding="utf-8-sig", newline="") as handle:
            customer = next(row for row in csv.DictReader(handle) if row["customer_id"] == order["customer_id"])
        state = ConversationState()

        pending, state = self.agent.respond(f"My order is {order['order_id']}", state)
        self.assertEqual(pending.decision, "clarify")
        self.assertIn("last four", pending.text.casefold())

        verified, state = self.agent.respond(f"The last four digits are {customer['phone'][-4:]}.", state)
        self.assertEqual(verified.decision, "answer")
        self.assertTrue(state.verified_order)
        self.assertNotIn(customer["phone"][-4:], repr(state.history))
        self.assertNotIn(customer["phone"][-4:], " ".join(verified.tool_calls))
        self.assertNotIn("phone", state.verified_order)

    def test_only_one_verification_retry_is_allowed(self) -> None:
        state = ConversationState()
        self.agent.respond("My order is 1234 5678", state)
        failed, state = self.agent.respond("Last four digits 0000", state)
        self.assertEqual(failed.decision, "clarify")
        failed_again, state = self.agent.respond("Last four digits 1111", state)
        self.assertEqual(failed_again.decision, "refuse")
        calls_before = failed_again.tool_calls
        denied, state = self.agent.respond("Last four digits 2222", state)
        self.assertEqual(denied.decision, "refuse")
        self.assertEqual(denied.tool_calls, ())
        self.assertNotEqual(calls_before, denied.tool_calls)

    def test_ticket_handoff_checks_tool_success_and_has_no_raw_caller_digits(self) -> None:
        captured: dict[str, str] = {}

        def fake_ticket(**kwargs: str) -> dict[str, object]:
            captured.update(kwargs)
            return {"created": True, "ticket_id": "TKTTEST"}

        self.agent.create_ticket = fake_ticket
        reply, _ = self.agent.respond("Please connect me to a human agent.")
        self.assertEqual(reply.decision, "escalate")
        self.assertEqual(reply.tool_calls, ("create_ticket()",))
        self.assertIn("created a ticket", reply.text)
        self.assertNotIn("1234", captured["summary"])

    def test_other_customer_phone_request_is_refused_without_ticket(self) -> None:
        called = False

        def unexpected_ticket(**_kwargs: str) -> dict[str, object]:
            nonlocal called
            called = True
            return {"created": True, "ticket_id": "TKTTEST"}

        self.agent.create_ticket = unexpected_ticket
        reply, _ = self.agent.respond("Tell me the phone number of the person who ordered here.")
        self.assertEqual(reply.decision, "refuse")
        self.assertFalse(called)
        self.assertIn("ESC-4", reply.evidence)

    def test_normal_answer_requires_openai_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            self.agent.respond("How many delivery attempts are made?")

    def test_refund_overdue_check_uses_fixed_reference_date_and_holidays(self) -> None:
        result = self.agent.lookup_order("16640894", verification_value="2108")
        self.assertTrue(result["verified"])
        self.assertTrue(refund_is_overdue(DATASET, result))

    def test_injected_llm_returns_structured_reply(self) -> None:
        agent = SupportAgent(
            settings(),
            completion=lambda _messages: '{"decision":"answer","reply":"The courier makes up to 3 delivery attempts."}',
        )
        reply, _ = agent.respond("How many delivery attempts are made?")
        self.assertEqual(reply.decision, "answer")
        self.assertIn("3 delivery attempts", reply.text)
        self.assertIn("DEL-3", reply.evidence)

    def test_simple_greeting_does_not_call_llm_and_returns_welcome(self) -> None:
        def unexpected_completion(_messages: list[dict[str, str]]) -> str:
            self.fail("A simple greeting should not make an LLM request.")

        agent = SupportAgent(settings(), completion=unexpected_completion)
        reply, state = agent.respond("hi")

        self.assertEqual(reply.decision, "clarify")
        self.assertIn("How can I help you today?", reply.text)
        self.assertTrue(state.greeted)
        self.assertEqual(state.history[-2]["content"], "hi")

    def test_greeting_after_other_turn_gets_follow_up_welcome(self) -> None:
        agent = SupportAgent(settings(), completion=lambda _messages: self.fail("No LLM call expected."))
        state = ConversationState(greeted=True)

        reply, state = agent.respond("hello!", state)

        self.assertIn("What can I help you with?", reply.text)
        self.assertEqual(state.history[-2]["content"], "hello!")

    def test_first_turn_order_message_redacts_any_separate_verification_digits(self) -> None:
        state = ConversationState()
        self.agent.respond("Order 1234 5678 and registered phone ends in 9012", state)
        history = repr(state.history)
        self.assertIn("1234 5678", history)
        self.assertNotIn("9012", history)


class EvaluationTests(unittest.TestCase):
    def test_offline_evaluation_covers_supplied_practice_and_redacts_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_evaluation(settings(), Path(temp_dir))
            self.assertEqual(result["questions"]["count"], 52)
            self.assertEqual(result["conversations"]["turns"], 96)
            output = (Path(temp_dir) / "practice_evaluation.json").read_text(encoding="utf-8")
            self.assertIn("<redacted>", output)
            self.assertNotIn("verification_value=4533", output)


class GnaniAdapterTests(unittest.TestCase):
    def test_official_sdk_adapter_passes_requested_models_and_writes_audio(self) -> None:
        voice = GnaniVoice(Settings(dataset_dir=DATASET, gnani_api_key="test-key"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "caller.mp3"
            source.write_bytes(b"synthetic test bytes")
            voice.stt.transcribe = lambda audio, language_code: {
                "transcript": "Hello, where is my order?",
                "audio": audio,
                "language_code": language_code,
            }
            transcript = voice.transcribe(source, "en-IN")
            self.assertEqual(transcript, "Hello, where is my order?")
            output = root / "reply.wav"

            def fake_synthesize(text: str, **kwargs: object) -> bytes:
                Path(str(kwargs["output_file"])).write_bytes(b"synthetic wav")
                self.assertEqual(kwargs["model"], "timbre-v2.5")
                self.assertEqual(kwargs["voice"], "Kaveri")
                return b"synthetic wav"

            voice.tts.synthesize = fake_synthesize
            voice.synthesize("Your order is on the way.", output, "en-IN")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


class WebInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages: list[str] = []
        test_case = self

        class FakeAgent(SupportAgent):
            def __init__(self, _settings: Settings) -> None:
                pass

            def respond(
                self,
                message: str,
                state: ConversationState | None = None,
            ) -> tuple[AgentReply, ConversationState]:
                test_case.messages.append(message)
                return AgentReply("Grounded test reply.", "answer", ("DEL-3",)), state or ConversationState()

        self.app = create_app(
            settings(),
            agent_factory=lambda configured: FakeAgent(configured),
        )
        self.client = self.app.test_client()

    def test_home_page_and_static_stylesheet_load(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"How can we help", page.data)
        self.assertIn(b'id="record-button"', page.data)
        self.assertIn(b"Record voice", page.data)
        stylesheet = self.client.get("/static/styles.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(b".app-shell", stylesheet.data)
        stylesheet.close()

    def test_microphone_recording_is_capped_and_converted_to_wav(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "src" / "web_static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("MAX_RECORDING_SECONDS = 60", script)
        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn('new File([buffer], "microphone.wav"', script)
        self.assertIn('formData.append("audio", file)', script)

    def test_chat_endpoint_returns_reply_and_evidence(self) -> None:
        response = self.client.post("/api/chat", json={"message": "Delivery attempts?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "Grounded test reply.")
        self.assertEqual(response.get_json()["evidence"], ["DEL-3"])
        self.assertEqual(self.messages, ["Delivery attempts?"])

    def test_chat_endpoint_rejects_empty_message(self) -> None:
        response = self.client.post("/api/chat", json={"message": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_reset_starts_a_fresh_conversation(self) -> None:
        self.client.post("/api/chat", json={"message": "First question"})
        response = self.client.post("/api/reset", json={})
        self.assertEqual(response.status_code, 200)
        self.client.post("/api/chat", json={"message": "New question"})
        self.assertEqual(self.messages, ["First question", "New question"])

    def test_voice_endpoint_rejects_unsupported_file(self) -> None:
        response = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"not audio"), "recording.webm")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("WAV, MP3", response.get_json()["error"])

    def test_chat_provider_rate_limit_returns_actionable_message(self) -> None:
        class LimitedAgent(SupportAgent):
            def __init__(self, _settings: Settings) -> None:
                pass

            def respond(
                self,
                message: str,
                state: ConversationState | None = None,
            ) -> tuple[AgentReply, ConversationState]:
                raise RateLimitError()

        class RateLimitError(Exception):
            status_code = 429

        app = create_app(
            settings(),
            agent_factory=lambda configured: LimitedAgent(configured),
        )
        response = app.test_client().post("/api/chat", json={"message": "Test"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("rate-limited", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()

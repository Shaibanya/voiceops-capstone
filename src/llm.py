from __future__ import annotations

import json
from typing import Any

from openai import BadRequestError


def complete_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except BadRequestError as exc:
        if "failed to generate json" not in str(exc).casefold():
            raise

        retry_messages = [message.copy() for message in messages]
        retry_messages[0]["content"] += (
            "\nThe prior response could not be formatted as JSON. Retry with exactly "
            "one syntactically valid JSON object and no Markdown or extra text."
        )
        response = client.chat.completions.create(
            model=model,
            messages=retry_messages,
            temperature=0.1,
        )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM returned an empty response.")
    content = content.strip()
    try:
        json.loads(content)
    except json.JSONDecodeError:
        repair_messages = [message.copy() for message in messages]
        repair_messages.extend([
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Answer the original request "
                    "again, following the system instructions, and return only one valid JSON "
                    "object with no Markdown or surrounding text."
                ),
            },
        ])
        repaired = client.chat.completions.create(
            model=model,
            messages=repair_messages,
            temperature=0,
        ).choices[0].message.content
        if not repaired:
            raise RuntimeError("The LLM returned an empty JSON repair response.")
        repaired = repaired.strip()
        try:
            json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise RuntimeError("The LLM response was not valid JSON after repair.") from exc
        return repaired
    return content

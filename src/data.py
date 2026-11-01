from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

NUMBER_WORDS = {
    "0": ("zero", "oh"),
    "1": ("one",),
    "2": ("two", "to", "too"),
    "3": ("three",),
    "4": ("four", "for"),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
}


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def find_products(dataset_dir: Path, query: str) -> list[dict[str, Any]]:
    path = dataset_dir / "data" / "products.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        products = list(csv.DictReader(handle))
    normalized_query = _normalized(query)
    matches = []
    for product in products:
        sku = product["sku"].casefold()
        title = _normalized(product["title"].split(",", 1)[0])
        prefix, _, suffix = sku.partition("-")
        spoken_sku_pattern = (
            rf"(?<![a-z]){re.escape(prefix)}[\s-]+"
            + r"[\s-]+".join(
                "(?:" + "|".join(re.escape(word) for word in NUMBER_WORDS[digit]) + ")"
                for digit in suffix
            )
            + r"(?![a-z])"
        )
        if (
            sku in query.casefold()
            or (title and title in normalized_query)
            or re.search(spoken_sku_pattern, query.casefold())
        ):
            matches.append(product)
    return matches


def product_query_context(products: list[dict[str, Any]], query: str) -> str:
    lowered = query.casefold()
    terms: list[str] = []
    for product in products:
        category = product.get("return_category", "")
        if any(term in lowered for term in ("return", "send back", "send it back", "wapas", "लौटा", "वापस")):
            terms.extend(("return window", "returns policy", category))
            if category == "fashion":
                terms.extend(("festive sale clothing", "return fee"))
            elif category == "grocery":
                terms.extend(("grocery nonreturnable", "damaged wrong missing"))
            elif category == "innerwear":
                terms.extend(("innerwear nonreturnable RET-5", "return eligibility"))
            elif category == "beauty_sealed":
                terms.extend(("beauty unopened seal", "return condition"))
        if any(term in lowered for term in ("warranty", "assured", "claim")):
            terms.extend(("warranty", "brand warranty", "ZentroMart Assured", product.get("fulfilled_by", "")))
        if any(term in lowered for term in ("crack", "damage", "damaged", "broken", "defect", "wrong")):
            try:
                if int(product.get("price_inr") or 0) > 10000:
                    terms.append("high value damaged claim priority resolution")
            except ValueError:
                pass
    return " ".join(term for term in terms if term)


def _holidays(dataset_dir: Path) -> set[date]:
    path = dataset_dir / "data" / "holidays.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {date.fromisoformat(row["date"]) for row in csv.DictReader(handle)}


def _add_business_days(start: date, count: int, holidays: set[date]) -> date:
    current = start
    added = 0
    while added < count:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holidays:
            added += 1
    return current


def _business_days_after(start: date, end: date, holidays: set[date]) -> int:
    current = start
    count = 0
    while current < end:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holidays:
            count += 1
    return count


def refund_is_overdue(dataset_dir: Path, order: dict[str, Any], reference_date: date = date(2026, 10, 20)) -> bool:
    timelines = (
        ("return_refund_initiated_on", "return_refund_completed_on", "return_refund_method"),
        ("rto_refund_initiated_on", "", "rto_refund_method"),
        ("cancellation_refund_initiated_on", "", "cancellation_refund_method"),
    )
    maximum_days = {
        "upi": 3,
        "card": 7,
        "netbanking": 5,
        "wallet": 0,
        "cod_wallet": 0,
        "cod_bank": 7,
    }
    holidays = _holidays(dataset_dir)
    for initiated_field, completed_field, method_field in timelines:
        initiated_text = order.get(initiated_field)
        if not initiated_text or (completed_field and order.get(completed_field)):
            continue
        method = str(order.get(method_field) or "").casefold()
        if method not in maximum_days:
            continue
        initiated = date.fromisoformat(str(initiated_text)[:10])
        due = _add_business_days(initiated, maximum_days[method], holidays)
        if reference_date > due and _business_days_after(due, reference_date, holidays) > 3:
            return True
    return False

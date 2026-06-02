from __future__ import annotations

import re
from dataclasses import dataclass


BLOCKED_PATTERNS = [
    r"(?i)\b(ignore|bypass|disable)\b.*\b(instructions|policy|security|rules)\b",
    r"(?i)\b(api[_ -]?key|password|secret|token)\b",
    r"(?i)<script\b",
    r"(?i)\b(drop table|delete from|shutdown)\b",
]


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    sanitized: str
    reason: str = ""


def validate_user_query(query: str, max_chars: int = 600) -> ValidationResult:
    cleaned = " ".join((query or "").strip().split())
    if not cleaned:
        return ValidationResult(False, "", "La consulta esta vacia.")
    if len(cleaned) > max_chars:
        return ValidationResult(False, cleaned[:max_chars], f"La consulta supera {max_chars} caracteres.")

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cleaned):
            return ValidationResult(False, cleaned, "La consulta contiene contenido no permitido por politicas de seguridad.")

    return ValidationResult(True, cleaned)


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))

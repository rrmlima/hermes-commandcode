"""Map Command Code failures onto Hermes failover reasons.

Consulted only for this provider. Returning ``None`` keeps Hermes' built-in verdict.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _status(error: Any, status_code: Any) -> Optional[int]:
    for candidate in (getattr(error, "status_code", None), status_code):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def classify_api_error(error: Any, *, status_code: Any = None, error_code: Any = None, message: Any = None,
                       body: Any = None, model: Any = None) -> Optional[Mapping[str, Any]]:
    code = _status(error, status_code)
    text = f"{message or ''} {body or ''} {error or ''}".lower()
    if code == 402 or "insufficient balance" in text or "insufficient_balance" in text:
        return {"reason": "billing", "should_fallback": True}
    if code == 401:
        return {"reason": "auth", "should_rotate_credential": True}
    if code == 403:
        # A plan without access to this model/route: the credential is fine, the request is not.
        return {"reason": "model_entitlement", "should_fallback": True}
    if code == 429:
        return {"reason": "rate_limit"}
    if code in (503, 529):
        return {"reason": "overloaded"}
    if code in (500, 502):
        return {"reason": "server_error"}
    if code in (408, 504):
        return {"reason": "timeout"}
    if code == 404:
        return {"reason": "model_not_found"}
    if code == 413:
        return {"reason": "payload_too_large"}
    if code == 400 and ("context" in text or "too long" in text or "maximum" in text):
        return {"reason": "context_overflow"}
    return None

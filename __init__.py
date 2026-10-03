"""Command Code for Hermes Agent — sign in with your Command Code account (no API key).

Registers the ``commandcode-oauth`` model provider. It speaks the same ``/alpha/generate``
endpoint the official Command Code CLI uses, so plans without Provider-API access (Go, free)
work, and it reuses the CLI's sign-in when there is one.

Everything goes through documented ``ProviderProfile`` seams — no Hermes core file is touched:

* ``create_client``      → :class:`client.CommandCodeClient` (the ``/alpha`` transport)
* ``fetch_models``       → the account's live catalog
* ``fetch_account_usage``→ credits and 5-hour / weekly windows for ``/usage``
* ``auth_handler`` / ``refresh_credential`` → ``hermes auth add|status|logout commandcode-oauth``
* ``classify_api_error`` → 402 billing, 401/403 auth, 429 rate limit … so failover works
"""

from __future__ import annotations

import datetime
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, List, Optional

from providers import register_provider
from providers.base import ProviderProfile

from .auth import PROVIDER, auth_handler, cli_token_or_empty, refresh_credential, whoami
from .client import CommandCodeClient
from .errors import classify_api_error
from .wire import ALPHA_ORIGIN

logger = logging.getLogger("hermes_commandcode")

__version__ = "0.1.0"

# Zero-cost entries go first so the picker's default costs nothing.
ZERO_COST_MODELS = ("poolside/laguna-s-2.1-free",)

# Offline fallback only — shown when the live catalog is unreachable.
FALLBACK_MODELS = (
    "poolside/laguna-s-2.1-free",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "meta/muse-spark-1.3",
    "moonshotai/Kimi-K3",
    "z-ai/glm-5.3-flash",
    "MiniMaxAI/MiniMax-M3",
    "Qwen/Qwen3.7-Max",
)


def _origin(base_url: Optional[str]) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw) if raw else None
    if parsed and parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ALPHA_ORIGIN


def _get_json(url: str, token: str, *, timeout: float) -> Optional[Any]:
    """Fail-open GET: any problem yields None."""
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "cli")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("GET %s failed: %s", url.split("?")[0], type(exc).__name__)
        return None


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class CommandCodeProfile(ProviderProfile):
    """Command Code over ``/alpha/generate``, authenticated with the account sign-in."""

    def create_client(self, **client_kwargs: Any) -> Any:
        # The pool hands the token in as ``api_key``; fall back to the CLI file so a fresh
        # CLI login works without re-running `hermes auth add`.
        return CommandCodeClient(token_provider=cli_token_or_empty, **client_kwargs)

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 8.0) -> list[str] | None:
        token = (api_key or "").strip() or cli_token_or_empty()
        if not token:
            return None
        payload = _get_json(f"{_origin(base_url)}/provider/v1/models", token, timeout=timeout)
        models: List[str] = []
        if isinstance(payload, dict):
            for item in payload.get("data") or payload.get("models") or []:
                model_id = item.get("id") if isinstance(item, dict) else item
                if isinstance(model_id, str) and model_id.strip() and model_id.strip() not in models:
                    models.append(model_id.strip())
        if not models:
            return None
        for name in reversed(ZERO_COST_MODELS):
            if name in models:
                models.remove(name)
                models.insert(0, name)
        return models

    def fetch_account_usage(self, *, api_key: str | None = None, base_url: str | None = None):
        token = (api_key or "").strip() or cli_token_or_empty()
        if not token:
            return None
        origin = _origin(base_url)
        identity = whoami(token, timeout=3.0) or {}
        org = identity.get("org_id") or ""
        suffix = f"?orgId={urllib.parse.quote(org)}" if org else ""
        credits = _get_json(f"{origin}/alpha/billing/credits{suffix}", token, timeout=3.0)
        if not isinstance(credits, dict):
            return None
        summary = _get_json(f"{origin}/alpha/usage/summary{suffix}", token, timeout=3.0)

        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

        windows = []
        for key, label in (("fiveHour", "5-hour"), ("weekly", "Weekly")):
            block = (credits.get("windowLimits") or {}).get(key) or {}
            used, cap = _num(block.get("used")), _num(block.get("cap"))
            if used is None or not cap:
                continue
            reset_ms = _num(block.get("resetAt"))
            windows.append(AccountUsageWindow(
                label=label,
                used_percent=max(0.0, min(100.0, used / cap * 100.0)),
                reset_at=(datetime.datetime.fromtimestamp(reset_ms / 1000.0, tz=datetime.timezone.utc)
                          if reset_ms and reset_ms > 0 else None),
                detail=f"${used:.2f} of ${cap:.2f}",
            ))
        details = []
        balance = credits.get("credits") or {}
        monthly, purchased = _num(balance.get("monthlyCredits")), _num(balance.get("purchasedCredits"))
        if monthly is not None:
            details.append(f"Monthly credits left: ${monthly:.2f}")
        if purchased:
            details.append(f"Purchased credits: ${purchased:.2f}")
        if isinstance(summary, dict) and _num(summary.get("totalCost")) is not None:
            calls = _num(summary.get("totalCount"))
            details.append(f"This billing period: ${summary['totalCost']:.2f}"
                           + (f" over {int(calls)} calls" if calls else ""))
        return AccountUsageSnapshot(
            provider=self.name, source="commandcode-billing",
            fetched_at=datetime.datetime.now(datetime.timezone.utc),
            title="Command Code limits", windows=tuple(windows), details=tuple(details), raw=credits,
        )


commandcode = CommandCodeProfile(
    name=PROVIDER,
    # Not aliased to "command-code": that string is the model-id vendor prefix.
    aliases=("commandcode-alpha", "commandcode-account"),
    display_name="Command Code (account sign-in)",
    description="Command Code with your account sign-in — the endpoint the Command Code CLI uses",
    signup_url="https://commandcode.ai/",
    api_mode="chat_completions",
    base_url=ALPHA_ORIGIN,
    # ``api_key`` so the model picker probes the live catalog; the credential itself comes from
    # the pool that ``auth_handler`` fills. The env var accepts the same bearer the CLI stores.
    auth_type="api_key",
    env_vars=("COMMANDCODE_CLI_TOKEN",),
    fallback_models=FALLBACK_MODELS,
    default_aux_model=ZERO_COST_MODELS[0],
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    classify_api_error=classify_api_error,
)

register_provider(commandcode)

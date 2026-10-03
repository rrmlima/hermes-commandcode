"""Wire format for Command Code's ``/alpha/generate`` NDJSON endpoint.

Pure functions: OpenAI-shaped request in, ``/alpha/generate`` body out; NDJSON events in,
OpenAI-shaped pieces out. No Hermes imports, so everything here is unit-testable alone.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger("hermes_commandcode.wire")

COMMANDCODE_GENERATE_URL = "https://api.commandcode.ai/alpha/generate"

# The wire protocol reports failures as an ``error`` event *inside a 200 stream*, and the
# status lives at the top level (``{"type":"error","statusCode":402,...}``). Carrying it on
# the exception is what lets ``agent.error_classifier`` map 402→billing, 401→auth,
# 429→rate_limit through the normal pipeline instead of a generic rejection.
class CommandCodeAPIError(RuntimeError):
    """Command Code transport/stream failure with an attached HTTP status code."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# The vendor registry marks these text-only: pixels sent to them are dropped server-side,
# so an attached image becomes a text placeholder instead of vanishing. Unknown ids stay
# image-capable (the registry's own fallback), which is why this is a deny-list.
_TEXT_ONLY_MODELS = frozenset({
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-fast",
    "zai-org/GLM-5.3", "zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast", "zai-org/GLM-5.1", "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.7", "minimax/minimax-m2.7-free", "MiniMaxAI/MiniMax-M2.5",
    "xiaomi/mimo-v2.5-pro", "Qwen/Qwen3.6-Max-Preview", "Qwen/Qwen3.7-Max",
    "meituan/LongCat-2.0:free", "stepfun/Step-3.5-Flash", "tencent/hy4-preview", "tencent/Hy3",
    "tencent/hy3-paid", "nvidia/nemotron-3-ultra-550b-a55b", "poolside/laguna-s-2.1-free",
    "inclusionai/ling-3.0-flash-free", "inclusionai/ling-3.0-flash-sante:free",
})


def _media_url(part: Dict[str, Any]) -> tuple[str, str]:
    """``(kind, url)`` for any shape a caller sends — flat ``image``, OpenAI's nested
    ``image_url``, or an Anthropic ``source`` block."""
    ptype = str(part.get("type", ""))
    inner = part.get("image_url") or part.get("video_url") or {}
    is_video = "video" in ptype
    is_media = is_video or "image" in ptype or bool(inner) or bool(part.get("source")) or bool(part.get("image"))
    if not is_media:
        # Unknown non-text part (file, audio, …): do not misrepresent it as an image.
        return "", ""
    kind = "video" if is_video else "image"
    url = ""
    if isinstance(inner, dict):
        url = str(inner.get("url", "") or "")
    elif isinstance(inner, str):
        url = inner
    if not url:
        url = str(part.get("image") or part.get("url") or (part.get("source") or {}).get("data", "") or "")
        src = part.get("source") or {}
        if url and src.get("data"):
            media_type = src.get("media_type") or "image/png"
            url = f"data:{media_type};base64,{url}"
    return kind, url


_URL_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _image_mime(url: str) -> str:
    """``image/png`` out of ``data:image/png;base64,…`` or an ``.png`` URL path; empty otherwise."""
    if url.startswith("data:") and "," in url:
        return url[5:].split(";", 1)[0].split(",", 1)[0]
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    for ext, mime in _URL_EXT_MIME.items():
        if path.endswith(ext):
            return mime
    return ""


def _normalize_media_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One non-text content part → the wire shape (``{"type": "image", "image": url}``).

    OpenAI's nested ``{"image_url": {"url": ...}}`` is rejected by the endpoint (400
    "expected string") and must be flattened. ``mimeType`` is what makes the endpoint
    actually process the pixels: the same part without it is accepted and silently
    ignored (verified against a solid-colour test image).
    """
    kind, url = _media_url(part)
    if not url:
        return None
    mime = str(part.get("mimeType") or part.get("mime_type") or "") or _image_mime(url)
    return {"type": kind, kind: url, "mimeType": mime} if mime else None


def _media_placeholder(part: Dict[str, Any]) -> str:
    kind, url = _media_url(part)
    return f"[{kind}: {_image_mime(url) or 'attached'}]"


def _format_messages_for_commandcode(
    messages: List[Dict[str, Any]], model: str = ""
) -> tuple[str, List[Dict[str, Any]]]:
    """Extract system prompt and convert messages to Command Code /alpha/generate wire format."""
    system_parts: List[str] = []
    wire_msgs: List[Dict[str, Any]] = []

    images_ok = (model or "").strip() not in _TEXT_ONLY_MODELS

    # Map assistant tool call IDs to names for tool results
    call_id_to_name: Dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        elif role == "user":
            parts_user: List[Dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts_user.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            parts_user.append({"type": "text", "text": text})
                        continue
                    media = _normalize_media_part(part) if images_ok else None
                    if media is not None:
                        parts_user.append(media)
                    else:
                        p_kind, _p_url = _media_url(part)
                        if not p_kind:
                            # Unknown part type: skip rather than claim it was an image.
                            continue
                        # Keep the turn honest: the model should see that something was
                        # attached even when the endpoint would silently drop the pixels.
                        parts_user.append({"type": "text", "text": _media_placeholder(part)})
            if not parts_user:
                parts_user.append({"type": "text", "text": ""})
            wire_msgs.append({
                "role": "user",
                "content": parts_user,
            })
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                call_id = tc.get("id") or str(uuid.uuid4())
                func = tc.get("function") or {}
                name = func.get("name", "tool")
                call_id_to_name[call_id] = name
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                parts.append({
                    "type": "tool-call",
                    "toolCallId": call_id,
                    "toolName": name,
                    "input": args,
                })
            if not parts:
                parts = [{"type": "text", "text": ""}]
            wire_msgs.append({
                "role": "assistant",
                "content": parts,
            })
        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            tool_name = call_id_to_name.get(call_id, "tool")
            val = content if isinstance(content, str) else json.dumps(content)
            wire_msgs.append({
                "role": "tool",
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                    "output": {"type": "text", "value": val},
                }],
            })

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, wire_msgs


def _format_tools_for_commandcode(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool definitions to Command Code input_schema format."""
    wire_tools: List[Dict[str, Any]] = []
    if not tools:
        return wire_tools

    for tool in tools:
        func = tool.get("function", {}) if tool.get("type") == "function" else tool
        name = func.get("name")
        if not name:
            continue
        params = func.get("parameters") or {"type": "object", "properties": {}}
        wire_tools.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": params,
        })
    return wire_tools


def _workspace_config(cwd: Optional[str] = None) -> Dict[str, Any]:
    working_dir = cwd or os.getcwd()
    return {
        "workingDir": working_dir,
        "isGitRepo": False,
        "currentBranch": "",
        "mainBranch": "",
        "gitStatus": "",
        "recentCommits": [],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        # The official client reports the real host; a hardcoded "linux" is wrong on
        # macOS/Windows and can steer the model's shell assumptions.
        "environment": f"{platform.system().lower()}-{platform.machine()}, Python {platform.python_version()}",
        "structure": [],
    }


# Mapping from dashed slugs to canonical wire model IDs accepted by /alpha/generate
COMMAND_CODE_MODEL_ALIASES: Dict[str, str] = {
    "deepseek-deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-deepseek-v4-flash-vision-exp": "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek-deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "meituan-LongCat-2.0:free": "meituan/LongCat-2.0:free",
    "meta-muse-spark-1.3-contributor": "meta/muse-spark-1.3-contributor",
    "MiniMaxAI-MiniMax-M3": "MiniMaxAI/MiniMax-M3",
    "moonshotai-Kimi-K3": "moonshotai/Kimi-K3",
    "poolside-laguna-s-2.1-free": "poolside/laguna-s-2.1-free",
    "Qwen-Qwen3.8-Max-0902": "Qwen/Qwen3.8-Max-0902",
    "xai-grok-4.5": "xai/grok-4.5",
    "xiaomi-mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "z-ai-glm-5.3-flash": "z-ai/glm-5.3-flash",
}


def canonical_commandcode_model_id(model_id: str) -> str:
    """Normalize model ID so dashed slugs or prefixed names map to canonical wire format."""
    clean = model_id.strip()
    if clean.startswith("command-code/"):
        clean = clean[len("command-code/"):]
    elif clean.startswith("commandcode-oauth/"):
        clean = clean[len("commandcode-oauth/"):]
    return COMMAND_CODE_MODEL_ALIASES.get(clean, clean)


ALPHA_ORIGIN = "https://api.commandcode.ai"
CLI_VERSION_HEADER = "0.52.1"


def generate_url(base_url: Optional[str]) -> str:
    """``{origin}/alpha/generate`` for the configured origin (a proxy may replace the public one)."""
    raw = (base_url or "").strip().rstrip("/")
    if not raw or "/provider/v1" in raw:
        raw = ALPHA_ORIGIN
    if raw.endswith("/alpha/generate"):
        return raw
    return f"{raw}/alpha/generate"


def build_request_body(api_kwargs: Dict[str, Any], model: str, *, cwd: Optional[str] = None) -> Dict[str, Any]:
    """OpenAI ``chat.completions.create`` kwargs → the ``/alpha/generate`` JSON body."""
    messages = api_kwargs.get("messages") or []
    tools = api_kwargs.get("tools")
    tool_choice = api_kwargs.get("tool_choice")
    if tool_choice == "none":
        # Honour "no tools": forwarding them anyway lets the model call something the
        # caller explicitly ruled out.
        tools = None
    elif isinstance(tool_choice, dict) or tool_choice == "required":
        # The wire has no forced-tool form; forward the tools rather than fail the turn.
        logger.debug("tool_choice=%r is not expressible on /alpha/generate; forwarding tools", tool_choice)
    system_prompt, wire_msgs = _format_messages_for_commandcode(messages, model)
    params: Dict[str, Any] = {
        "model": model,
        "messages": wire_msgs,
        "tools": _format_tools_for_commandcode(tools),
        "system": system_prompt,
        "max_tokens": api_kwargs.get("max_tokens") or api_kwargs.get("max_completion_tokens") or 4096,
        "stream": True,
    }
    temperature = api_kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        params["temperature"] = temperature
    return {
        "config": _workspace_config(cwd),
        "memory": "",
        "taste": None,
        "skills": None,
        "permissionMode": "standard",
        "mode": "agent",
        "params": params,
    }


def request_headers(token: str) -> Dict[str, str]:
    """Headers the official CLI sends; a fresh ``x-session-id`` per request (the cache is prefix-keyed)."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "cli",
        "x-command-code-version": CLI_VERSION_HEADER,
        "x-cli-environment": "production",
        "x-taste-learning": "false",
        "x-co-flag": "false",
        "x-session-id": str(uuid.uuid4()),
    }


def error_from_event(event: Dict[str, Any]) -> CommandCodeAPIError:
    """An in-stream ``{"type": "error", "statusCode": 402, ...}`` as an exception carrying the status.

    The status lives at the TOP level of the event. Carrying it is what lets Hermes map
    402 → billing, 401 → auth, 429 → rate_limit instead of a generic failure.
    """
    status = event.get("statusCode")
    status = int(status) if isinstance(status, (int, str)) and str(status).isdigit() else None
    err_obj = event.get("error")
    if isinstance(err_obj, dict):
        status = status or (int(err_obj["statusCode"]) if str(err_obj.get("statusCode", "")).isdigit() else None)
        message = err_obj.get("message") or err_obj.get("type") or "Command Code generation error"
    else:
        message = event.get("message") or (str(err_obj) if err_obj else "") or "Command Code generation error"
    text = f"Command Code stream error (HTTP {status}): {message}" if status else f"Command Code stream error: {message}"
    return CommandCodeAPIError(text, status_code=status)


def iter_events(lines: Iterable[bytes]) -> Iterator[Dict[str, Any]]:
    """Decode NDJSON lines into events; an ``error`` event raises instead of being yielded."""
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            raise error_from_event(event)
        yield event


def usage_from_event(event: Dict[str, Any]) -> Optional[SimpleNamespace]:
    """OpenAI-shaped usage from a ``finish``/``finish-step`` event, or None when it carries none.

    The relay reports the cached share under ``inputTokenDetails.cacheReadTokens`` and bills it
    separately; mapping it to ``prompt_tokens_details.cached_tokens`` is what makes Hermes show
    the cache hits.
    """
    raw = event.get("totalUsage") or event.get("usage") or {}
    if not isinstance(raw, dict) or not raw:
        return None
    prompt = int(raw.get("inputTokens") or raw.get("prompt_tokens") or 0)
    completion = int(raw.get("outputTokens") or raw.get("completion_tokens") or 0)
    details = raw.get("inputTokenDetails") or {}
    cache_read = int(details.get("cacheReadTokens") or raw.get("cacheReadTokens") or raw.get("cachedInputTokens") or 0)
    cache_write = int(details.get("cacheWriteTokens") or raw.get("cacheWriteTokens") or 0)
    if not (prompt or completion or cache_read or cache_write):
        return None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read, cache_write_tokens=cache_write),
    )

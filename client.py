"""OpenAI-client-shaped transport over Command Code's ``/alpha/generate``.

Hermes calls ``client.chat.completions.create(**kwargs)`` in three ways and this client
serves all three from one object, which is why it declares both opt-outs below:

* main loop, ``stream=True``  → iterate OpenAI-style chunks as tokens arrive;
* sync auxiliary, no stream   → read ``.choices`` off the result;
* async auxiliary (vision, title, compression) → ``await`` the same call.

Without ``HERMES_SKIP_ASYNC_WRAP`` the auxiliary client would rebuild an ``AsyncOpenAI``
against ``base_url`` — an endpoint that does not accept the CLI bearer — so the async
path must get this object back unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

from .wire import (
    ALPHA_ORIGIN,
    CommandCodeAPIError,
    build_request_body,
    canonical_commandcode_model_id,
    generate_url,
    iter_events,
    request_headers,
    usage_from_event,
)

logger = logging.getLogger("hermes_commandcode.client")

DEFAULT_TIMEOUT_S = 180.0


def coerce_timeout(value: Any, default: float = DEFAULT_TIMEOUT_S) -> float:
    """Seconds from a float, an ``httpx.Timeout``-like object, or nothing."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    for attr in ("read", "timeout", "connect"):
        inner = getattr(value, attr, None)
        if isinstance(inner, (int, float)) and inner > 0:
            return float(inner)
    return default


def _chunk(response_id: str, model: str, *, content: Optional[str] = None, reasoning: Optional[str] = None,
           tool_calls: Optional[List[Any]] = None, finish_reason: Optional[str] = None,
           usage: Any = None) -> SimpleNamespace:
    delta = SimpleNamespace(role="assistant", content=content, reasoning_content=reasoning,
                            reasoning=reasoning, tool_calls=tool_calls)
    choices = [] if (usage is not None and finish_reason is None and content is None
                     and reasoning is None and tool_calls is None) else [
        SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(id=response_id, object="chat.completion.chunk", model=model,
                           choices=choices, usage=usage)


def _tool_call(event: Dict[str, Any], index: int) -> SimpleNamespace:
    raw_input = event.get("input", {})
    arguments = raw_input if isinstance(raw_input, str) else json.dumps(raw_input)
    return SimpleNamespace(index=index, id=event.get("toolCallId") or f"call_{uuid.uuid4().hex[:12]}",
                           type="function",
                           function=SimpleNamespace(name=event.get("toolName") or "tool", arguments=arguments))


class _LazyResult:
    """Result of a non-streaming ``create``: runs on first attribute access, or when awaited.

    Lazy so that ``await client.chat.completions.create(...)`` performs the HTTP call in a
    worker thread instead of blocking the event loop at call time.
    """

    def __init__(self, run: Callable[[], Any]) -> None:
        self._run = run
        self._value: Any = None
        self._lock = threading.Lock()

    def _resolve(self) -> Any:
        with self._lock:
            if self._value is None:
                self._value = self._run()
            return self._value

    def __await__(self):
        return asyncio.to_thread(self._resolve).__await__()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


class _StreamResult:
    """Result of a streaming ``create``: a chunk iterator, or the aggregated response when awaited.

    Deliberately has no ``choices`` attribute: Hermes treats an object with ``choices`` as an
    already-complete response and would skip the stream.
    """

    def __init__(self, stream: Callable[[], Iterator[Any]], complete: Callable[[], Any]) -> None:
        self._stream = stream
        self._complete = complete

    def __iter__(self) -> Iterator[Any]:
        return self._stream()

    def __await__(self):
        return asyncio.to_thread(self._complete).__await__()


class CommandCodeClient:
    """The ``/alpha/generate`` transport, shaped like ``openai.OpenAI`` for Hermes."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: Any = None, token_provider: Optional[Callable[[], str]] = None, **_: Any) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = str(base_url or ALPHA_ORIGIN)
        self._url = generate_url(self.base_url)
        self._timeout = coerce_timeout(timeout)
        self._token_provider = token_provider
        self._open: set = set()
        self._open_lock = threading.Lock()
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    # ── OpenAI surface ───────────────────────────────────────────────────────
    def _create(self, *, stream: bool = False, **kwargs: Any) -> Any:
        if stream:
            return _StreamResult(lambda: self._iter_chunks(kwargs), lambda: self._complete(kwargs))
        return _LazyResult(lambda: self._complete(kwargs))

    def close(self) -> None:
        self.is_closed = True
        self.cancel()

    def cancel(self) -> None:
        """Abort in-flight requests (called from the interrupting thread)."""
        with self._open_lock:
            responses = tuple(self._open)
        for resp in responses:
            try:
                resp.close()
            except Exception:  # pragma: no cover - best effort
                pass

    # ── transport ────────────────────────────────────────────────────────────
    def _token(self) -> str:
        token = self.api_key
        if not token and self._token_provider is not None:
            token = (self._token_provider() or "").strip()
        if not token:
            raise CommandCodeAPIError(
                "No Command Code credential: run `hermes auth add commandcode-oauth` "
                "(or sign in with the Command Code CLI).", status_code=401)
        return token

    @contextmanager
    def _post(self, kwargs: Dict[str, Any]):
        model = canonical_commandcode_model_id(str(kwargs.get("model") or ""))
        if not model:
            raise CommandCodeAPIError("Command Code request is missing a model", status_code=400)
        body = json.dumps(build_request_body(kwargs, model)).encode("utf-8")
        req = urllib.request.Request(self._url, data=body, method="POST")
        for key, value in request_headers(self._token()).items():
            req.add_header(key, value)
        timeout = coerce_timeout(kwargs.get("timeout"), self._timeout)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise CommandCodeAPIError(f"Command Code HTTP {exc.code}: {detail or exc.reason}",
                                      status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise CommandCodeAPIError(f"Command Code unreachable: {exc.reason}", status_code=503) from exc
        with self._open_lock:
            self._open.add(resp)
        try:
            yield model, resp
        finally:
            with self._open_lock:
                self._open.discard(resp)
            resp.close()

    def _iter_chunks(self, kwargs: Dict[str, Any]) -> Iterator[Any]:
        response_id = f"cmdcode-{uuid.uuid4().hex[:12]}"
        produced = finished = False
        tool_calls: List[Any] = []
        with self._post(kwargs) as (model, resp):
            for event in iter_events(resp):
                kind = event.get("type")
                if kind == "text-delta" and event.get("text"):
                    produced = True
                    yield _chunk(response_id, model, content=event["text"])
                elif kind == "reasoning-delta" and event.get("text"):
                    yield _chunk(response_id, model, reasoning=event["text"])
                elif kind == "tool-call":
                    produced = True
                    tool_calls.append(_tool_call(event, len(tool_calls)))
                    yield _chunk(response_id, model, tool_calls=[tool_calls[-1]])
                elif kind in ("finish", "finish-step"):
                    finished = True
                    usage = usage_from_event(event)
                    if kind == "finish":
                        reason = "tool_calls" if tool_calls else (event.get("finishReason") or "stop")
                        yield _chunk(response_id, model, finish_reason=reason, usage=usage)
                    elif usage is not None:
                        yield _chunk(response_id, model, usage=usage)
        if not produced or not finished:
            # A 200 stream that stopped early: an empty turn is worse than an error.
            raise CommandCodeAPIError(
                f"Command Code stream ended without an answer (output={produced}, finished={finished})",
                status_code=502)

    def _complete(self, kwargs: Dict[str, Any]) -> SimpleNamespace:
        content: List[str] = []
        reasoning: List[str] = []
        tool_calls: List[Any] = []
        finish_reason = "stop"
        usage = None
        model = ""
        response_id = f"cmdcode-{uuid.uuid4().hex[:12]}"
        for chunk in self._iter_chunks(kwargs):
            model = chunk.model
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    content.append(delta.content)
                if delta.reasoning_content:
                    reasoning.append(delta.reasoning_content)
                for call in delta.tool_calls or ():
                    tool_calls.append(SimpleNamespace(id=call.id, type="function", function=call.function))
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        message = SimpleNamespace(role="assistant", content="".join(content) or None,
                                  reasoning_content="".join(reasoning) or None,
                                  tool_calls=tool_calls or None)
        return SimpleNamespace(
            id=response_id, object="chat.completion", model=model,
            choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
            usage=usage or SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                                           prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        )

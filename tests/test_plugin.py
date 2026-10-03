"""Hermetic tests: no network, no real Hermes home, no real Command Code account.

A local HTTP server plays ``/alpha/generate``. Run from the repository root:

    HERMES_TREE=/path/to/hermes-agent python -m pytest -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_TREE = os.environ.get("HERMES_TREE")
if HERMES_TREE:
    sys.path.insert(0, HERMES_TREE)

# Import the plugin as a package (it uses relative imports, like Hermes loads it).
_spec = importlib.util.spec_from_file_location("hermes_commandcode", ROOT / "__init__.py",
                                               submodule_search_locations=[str(ROOT)])
needs_hermes = pytest.mark.skipif(not HERMES_TREE, reason="set HERMES_TREE to a hermes-agent checkout")


def _load_submodule(name):
    pkg = sys.modules.get("hermes_commandcode")
    if pkg is None:
        pkg = importlib.util.module_from_spec(_spec)
        pkg.__path__ = [str(ROOT)]
        sys.modules["hermes_commandcode"] = pkg
    spec = importlib.util.spec_from_file_location(f"hermes_commandcode.{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wire = _load_submodule("wire")
errors = _load_submodule("errors")
auth = _load_submodule("auth")
client_mod = _load_submodule("client")


# ── fake /alpha/generate ─────────────────────────────────────────────────────

class _Relay:
    def __init__(self, events, status=200):
        self.events, self.status, self.requests = events, status, []
        relay = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                relay.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                self.send_response(relay.status)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for event in relay.events:
                    self.wfile.write((json.dumps(event) + "\n").encode())

            def log_message(self, *_):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def relay():
    made = []

    def make(events, status=200):
        r = _Relay(events, status)
        made.append(r)
        return r

    yield make
    for r in made:
        r.close()


def _client(r, **kw):
    return client_mod.CommandCodeClient(api_key="tok", base_url=r.url, **kw)


TEXT = [{"type": "start"}, {"type": "text-delta", "text": "Hel"}, {"type": "text-delta", "text": "lo"},
        {"type": "finish", "finishReason": "stop",
         "totalUsage": {"inputTokens": 100, "outputTokens": 5,
                        "inputTokenDetails": {"cacheReadTokens": 90, "cacheWriteTokens": 3}}}]


# ── wire format ──────────────────────────────────────────────────────────────

def test_image_is_flattened_with_mime_type():
    body = wire.build_request_body({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}, "meta/muse-spark-1.3")
    parts = body["params"]["messages"][0]["content"]
    assert parts[1] == {"type": "image", "image": "data:image/png;base64,AAAA", "mimeType": "image/png"}


def test_anthropic_source_block_becomes_data_uri():
    body = wire.build_request_body({"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBBB"}}]}]},
        "meta/muse-spark-1.3")
    part = body["params"]["messages"][0]["content"][0]
    assert part["image"] == "data:image/jpeg;base64,BBBB" and part["mimeType"] == "image/jpeg"


def test_text_only_model_gets_placeholder_instead_of_silent_drop():
    body = wire.build_request_body({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}, "deepseek/deepseek-v4-pro")
    assert body["params"]["messages"][0]["content"] == [{"type": "text", "text": "[image: image/png]"}]


def test_tool_choice_none_drops_tools_and_temperature_is_forwarded():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    body = wire.build_request_body({"messages": [], "tools": tools, "tool_choice": "none", "temperature": 0.2}, "m")
    assert body["params"]["tools"] == [] and body["params"]["temperature"] == 0.2
    body = wire.build_request_body({"messages": [], "tools": tools, "temperature": True}, "m")
    assert body["params"]["tools"][0]["name"] == "f" and "temperature" not in body["params"]


def test_environment_reports_the_real_host():
    import platform
    env = wire.build_request_body({"messages": []}, "m")["config"]["environment"]
    assert env.startswith(platform.system().lower())


def test_tool_round_trip_keeps_call_ids_and_names():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"p\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    wire_msgs = wire.build_request_body({"messages": msgs}, "m")["params"]["messages"]
    assert wire_msgs[0]["content"][0] == {"type": "tool-call", "toolCallId": "c1", "toolName": "read", "input": {"p": 1}}
    assert wire_msgs[1]["content"][0]["toolName"] == "read"


def test_generate_url_honours_a_proxy_and_ignores_provider_api_path():
    assert wire.generate_url("http://proxy:8080/") == "http://proxy:8080/alpha/generate"
    assert wire.generate_url("https://api.commandcode.ai/provider/v1") == "https://api.commandcode.ai/alpha/generate"


# ── transport ────────────────────────────────────────────────────────────────

def test_stream_yields_text_then_finish_with_cache_usage(relay):
    r = relay(TEXT)
    chunks = list(_client(r).chat.completions.create(model="m", messages=[], stream=True))
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello"
    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop"
    assert last.usage.prompt_tokens_details.cached_tokens == 90
    assert r.requests[0]["path"] == "/alpha/generate"
    assert r.requests[0]["headers"]["Authorization"] == "Bearer tok"


def test_non_stream_result_is_a_complete_response(relay):
    r = relay(TEXT)
    resp = _client(r).chat.completions.create(model="m", messages=[])
    assert resp.choices[0].message.content == "Hello" and resp.usage.total_tokens == 105


def test_async_auxiliary_path_awaits_the_same_transport(relay):
    r = relay(TEXT)
    c = _client(r)

    async def go():
        return await c.chat.completions.create(model="m", messages=[])

    assert asyncio.run(go()).choices[0].message.content == "Hello"
    assert r.requests[0]["path"] == "/alpha/generate"


def test_async_stream_request_resolves_to_a_complete_response(relay):
    r = relay(TEXT)
    c = _client(r)

    async def go():
        return await c.chat.completions.create(model="m", messages=[], stream=True)

    resp = asyncio.run(go())
    assert hasattr(resp, "choices") and resp.choices[0].message.content == "Hello"


def test_stream_object_does_not_look_complete():
    # Hermes treats an object with `choices` as already complete and would skip streaming.
    c = client_mod.CommandCodeClient(api_key="tok")
    assert not hasattr(c.chat.completions.create(model="m", messages=[], stream=True), "choices")


def test_tool_call_sets_finish_reason(relay):
    r = relay([{"type": "tool-call", "toolCallId": "t1", "toolName": "ls", "input": {"d": "."}},
               {"type": "finish", "finishReason": "tool-calls"}])
    resp = _client(r).chat.completions.create(model="m", messages=[])
    call = resp.choices[0].message.tool_calls[0]
    assert resp.choices[0].finish_reason == "tool_calls"
    assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("t1", "ls", {"d": "."})


def test_empty_stream_fails_closed(relay):
    r = relay([{"type": "start"}, {"type": "finish", "finishReason": "stop"}])
    with pytest.raises(wire.CommandCodeAPIError) as exc:
        _client(r).chat.completions.create(model="m", messages=[]).choices
    assert exc.value.status_code == 502


def test_in_stream_402_carries_status_and_classifies_as_billing(relay):
    r = relay([{"type": "start"}, {"type": "error", "statusCode": 402, "error": "Insufficient Balance"}])
    with pytest.raises(wire.CommandCodeAPIError) as exc:
        _client(r).chat.completions.create(model="m", messages=[]).choices
    assert exc.value.status_code == 402
    assert errors.classify_api_error(exc.value)["reason"] == "billing"


def test_http_error_carries_status(relay):
    r = relay([], status=429)
    with pytest.raises(wire.CommandCodeAPIError) as exc:
        _client(r).chat.completions.create(model="m", messages=[]).choices
    assert exc.value.status_code == 429


def test_missing_credential_is_a_401_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = client_mod.CommandCodeClient(token_provider=auth.cli_token_or_empty)
    with pytest.raises(wire.CommandCodeAPIError) as exc:
        c.chat.completions.create(model="m", messages=[]).choices
    assert exc.value.status_code == 401


def test_classifier_table():
    cases = {401: "auth", 403: "model_entitlement", 429: "rate_limit", 503: "overloaded", 502: "server_error",
             504: "timeout", 404: "model_not_found", 413: "payload_too_large"}
    for code, reason in cases.items():
        assert errors.classify_api_error(SimpleNamespace(status_code=code))["reason"] == reason
    assert errors.classify_api_error(SimpleNamespace(status_code=418)) is None


# ── sign-in ──────────────────────────────────────────────────────────────────

def test_busy_callback_port_fails_fast_instead_of_binding_elsewhere():
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(auth.CommandCodeAuthError, match=str(port)):
            auth.start_callback_server("s", port=port)
    finally:
        blocker.close()


def test_callback_listens_before_the_url_exists_and_checks_state():
    import urllib.request
    server, thread, result = auth.start_callback_server("good", port=0)
    port = server.server_address[1]

    def post(payload):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/callback", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=2).status
        except urllib.error.HTTPError as exc:
            return exc.code

    try:
        assert post({"state": "bad", "apiKey": "k"}) == 400
        result["error"] = None
        assert post({"state": "good", "apiKey": "k"}) == 200
        assert auth.wait_for_callback(server, thread, result, timeout=5)["apiKey"] == "k"
    finally:
        server.server_close()


def test_cli_grant_round_trip_is_private(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    auth.write_cli_token("secret", user_name="r")
    assert auth.read_cli_token() == "secret"
    assert oct(auth.cli_auth_path().stat().st_mode & 0o777) == "0o600"


@needs_hermes
def test_dead_grant_marks_row_for_relogin(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(auth, "whoami", lambda token, **_: None)
    from hermes_cli.auth_constants import AuthError
    with pytest.raises(AuthError) as exc:
        auth.refresh_credential(SimpleNamespace(access_token="old"))
    assert exc.value.relogin_required


@needs_hermes
def test_plugin_registers_through_hermes_discovery(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    target = home / "plugins" / "model-providers" / "commandcode-oauth"
    target.parent.mkdir(parents=True)
    target.symlink_to(ROOT, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from providers import get_provider_profile
    import providers
    if hasattr(providers, "_reset_for_tests"):
        providers._reset_for_tests()
    profile = get_provider_profile("commandcode-oauth")
    assert profile is not None and profile.auth_handler is not None
    assert get_provider_profile("commandcode-alpha").name == "commandcode-oauth"
    client = profile.create_client(api_key="tok", base_url=profile.base_url)
    assert client.HERMES_SKIP_ASYNC_WRAP and client.HERMES_SKIP_TRANSPORT_WRAP

"""Command Code sign-in for Hermes.

Two sources, in order:

1. **Import** the grant the official Command Code CLI already stored in
   ``~/.commandcode/auth.json`` (validated against ``/alpha/whoami``).
2. **Browser hand-off**: the studio page POSTs the grant to ``127.0.0.1:5959/callback``.
   The listener is bound and serving *before* the URL is printed or the browser opens, and a
   busy port is an immediate error — the studio only knows the port the URL names.

The grant lands in Hermes' credential pool; it is also written back to the CLI file so the
two tools share one login. Tokens are never logged.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import quote, urlparse

logger = logging.getLogger("hermes_commandcode.auth")

PROVIDER = "commandcode-oauth"
WHOAMI_URL = "https://api.commandcode.ai/alpha/whoami"
STUDIO_URL = "https://commandcode.ai"
CALLBACK_PORT = 5959
POOL_SOURCE = "manual:commandcode_cli"
_ALLOWED_ORIGINS = ("https://commandcode.ai", "https://www.commandcode.ai")


class CommandCodeAuthError(RuntimeError):
    """A credential problem worth showing verbatim. Never carries the token."""


def cli_auth_path() -> Path:
    return Path(os.environ.get("HOME") or Path.home()) / ".commandcode" / "auth.json"


# ── the CLI's credential file ────────────────────────────────────────────────

def read_cli_token() -> str:
    path = cli_auth_path()
    if not path.exists():
        raise CommandCodeAuthError(f"No Command Code CLI sign-in at {path}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommandCodeAuthError(f"Could not read {path}: {exc}") from exc
    token = str((data or {}).get("apiKey") or "").strip() if isinstance(data, dict) else ""
    if not token:
        raise CommandCodeAuthError(f"{path} has no apiKey.")
    return token


def cli_token_or_empty() -> str:
    try:
        return read_cli_token()
    except CommandCodeAuthError:
        return ""


def write_cli_token(token: str, *, user_id: str = "", user_name: str = "") -> None:
    """Atomic, 0600 — the same file and shape the CLI writes."""
    path = cli_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"apiKey": token, "userId": user_id, "userName": user_name, "keyName": "hermes",
               "authenticatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())}
    tmp = path.with_name(f".auth.json.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def whoami(token: str, *, timeout: float = 10.0) -> Optional[Dict[str, str]]:
    """``{"user_id", "user_name", "org_id"}`` when the grant is accepted, else None."""
    if not (token or "").strip():
        return None
    req = urllib.request.Request(WHOAMI_URL)
    req.add_header("Authorization", f"Bearer {token.strip()}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "cli")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("whoami failed: %s", type(exc).__name__)
        return None
    user = data.get("user") if isinstance(data, dict) else None
    user = user if isinstance(user, dict) else {}
    org = ""
    for key in ("organizationId", "orgId", "organization_id"):
        value = (data or {}).get(key) if isinstance(data, dict) else None
        if isinstance(value, str) and value.strip():
            org = value.strip()
            break
    return {"user_id": str(user.get("id") or ""), "user_name": str(user.get("userName") or ""), "org_id": org}


# ── browser hand-off ─────────────────────────────────────────────────────────

def _handler(expected_state: str):
    result: Dict[str, Any] = {"payload": None, "error": None}

    class _Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: Dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            origin = self.headers.get("Origin") or ""
            if origin in _ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._reply(204 if urlparse(self.path).path == "/callback" else 404, {})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/callback":
                self._reply(404, {"success": False})
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 64 * 1024)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, OSError):
                self._reply(400, {"success": False, "error": "bad request"})
                return
            if not isinstance(payload, dict) or not secrets.compare_digest(
                    str(payload.get("state") or ""), expected_state):
                result["error"] = "state mismatch"
                self._reply(400, {"success": False, "error": "state mismatch"})
                return
            if not str(payload.get("apiKey") or "").strip():
                result["error"] = "no apiKey in callback"
                self._reply(400, {"success": False, "error": "missing apiKey"})
                return
            result["payload"] = payload
            self._reply(200, {"success": True})

        def log_message(self, *_: Any) -> None:
            return

    return _Handler, result


def start_callback_server(state: str, *, port: int = CALLBACK_PORT) -> Tuple[HTTPServer, threading.Thread, Dict[str, Any]]:
    """Bind and serve FIRST; the URL handed to the browser names this exact port."""
    handler, result = _handler(state)
    try:
        server = HTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        raise CommandCodeAuthError(
            f"Port {port} is already in use, so the Command Code sign-in cannot reach Hermes. "
            f"Stop whatever listens on 127.0.0.1:{port} (often another `cmd login`) and retry — "
            "or sign in with the Command Code CLI first and Hermes will import that grant."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return server, thread, result


def wait_for_callback(server: HTTPServer, thread: threading.Thread, result: Dict[str, Any],
                      *, timeout: float = 180.0) -> Dict[str, Any]:
    deadline = time.monotonic() + max(5.0, timeout)
    try:
        while time.monotonic() < deadline:
            if result["error"]:
                raise CommandCodeAuthError(f"Command Code sign-in failed: {result['error']}")
            if result["payload"]:
                return result["payload"]
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise CommandCodeAuthError("Timed out waiting for the browser sign-in.")


def login(*, open_browser: bool = True, timeout: float = 180.0, force_browser: bool = False) -> Dict[str, str]:
    """Import the CLI grant when it is valid; otherwise run the browser hand-off."""
    if not force_browser:
        token = cli_token_or_empty()
        identity = whoami(token) if token else None
        if identity is not None:
            print(f"Imported the Command Code CLI sign-in ({cli_auth_path()}).")
            return {"token": token, "source": "commandcode-cli", **identity}

    state = secrets.token_urlsafe(32)
    server, thread, result = start_callback_server(state)
    callback = f"http://127.0.0.1:{CALLBACK_PORT}/callback"
    url = f"{STUDIO_URL}/studio/auth/cli?callback={quote(callback, safe='')}&state={state}"
    print("Sign in to Command Code in your browser:\n  " + url + "\n")
    print(f"(Remote machine? Forward the port first: ssh -L {CALLBACK_PORT}:127.0.0.1:{CALLBACK_PORT} <host>)")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    payload = wait_for_callback(server, thread, result, timeout=timeout)
    token = str(payload["apiKey"]).strip()
    user_id, user_name = str(payload.get("userId") or ""), str(payload.get("userName") or "")
    try:
        write_cli_token(token, user_id=user_id, user_name=user_name)
    except OSError as exc:
        logger.debug("could not share the grant with the CLI: %s", exc)
    return {"token": token, "source": "commandcode-studio", "user_id": user_id, "user_name": user_name, "org_id": ""}


# ── ProviderProfile hooks ────────────────────────────────────────────────────

def auth_handler(action: str, args: Any) -> bool:
    """``hermes auth add|status|logout commandcode-oauth``. ``refresh`` goes to the pool."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

    pool = load_pool(PROVIDER)
    if action == "add":
        creds = login(open_browser=not getattr(args, "no_browser", False),
                      timeout=float(getattr(args, "timeout", None) or 180.0),
                      force_browser=bool(getattr(args, "force", False)))
        for index in range(len(pool.entries()), 0, -1):
            entry = pool.entries()[index - 1]
            if entry.source == POOL_SOURCE:
                pool.remove_index(index)
        entry = pool.add_entry(PooledCredential(
            provider=PROVIDER, id=uuid.uuid4().hex[:6],
            label=creds.get("user_name") or creds.get("user_id") or "Command Code",
            auth_type=AUTH_TYPE_OAUTH, priority=0, source=POOL_SOURCE, access_token=creds["token"],
            extra={"commandcode": {"source": creds.get("source", ""), "org_id": creds.get("org_id", "")}},
        ))
        print(f"Signed in to Command Code as {entry.label}.")
        return True
    if action == "status":
        entries = [e for e in pool.entries() if (e.access_token or "").strip()]
        if not entries:
            print(f"{PROVIDER}: logged out — run `hermes auth add {PROVIDER}`")
            return True
        identity = whoami(entries[0].access_token)
        if identity is None:
            print(f"{PROVIDER}: credential present but rejected — run `hermes auth add {PROVIDER}`")
        else:
            print(f"{PROVIDER}: logged in as {identity['user_name'] or identity['user_id'] or 'unknown'}"
                  f" ({len(entries)} credential(s))")
        return True
    if action == "logout":
        count = len(pool.entries())
        for index in range(count, 0, -1):
            pool.remove_index(index)
        print(f"Logged out of {PROVIDER} ({count} credential(s) removed).")
        return True
    return False


def refresh_credential(entry: Any) -> Optional[Mapping[str, Any]]:
    """The grant has no refresh token; the CLI file is the source of truth.

    If the CLI signed in again since, adopt that grant; if the stored one still validates, keep it.
    A dead grant raises ``relogin_required`` so the pool marks the row DEAD instead of retrying.
    """
    for candidate in (cli_token_or_empty(), str(getattr(entry, "access_token", "") or "")):
        if candidate and whoami(candidate) is not None:
            return {"access_token": candidate}
    from hermes_cli.auth_constants import AuthError

    raise AuthError(f"Command Code sign-in is no longer valid — run `hermes auth add {PROVIDER}`.",
                    provider=PROVIDER, relogin_required=True)

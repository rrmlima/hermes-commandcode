# hermes-commandcode

Use **Command Code** in [Hermes Agent](https://github.com/NousResearch/hermes-agent) with your
Command Code **account sign-in** — no API key. Works on plans without Provider-API access (Go,
free), which get `403 Your plan doesn't include API access` from the built-in `commandcode`
provider.

It talks to the same `/alpha/generate` endpoint the official Command Code CLI uses, and it reuses
the CLI's sign-in when you already have one.

## Install

```bash
hermes plugins install rrmlima/hermes-commandcode --enable
hermes auth add commandcode-oauth     # imports ~/.commandcode/auth.json, or opens the browser
hermes model                          # pick "Command Code (account sign-in)"
```

Pin an exact commit for reproducible installs:

```bash
hermes plugins install rrmlima/hermes-commandcode --ref <40-char-sha> --enable
```

Remote machine (SSH)? The browser sign-in posts to `127.0.0.1:5959` on the machine running
Hermes. Either sign in with the Command Code CLI there first, or forward the port:
`ssh -L 5959:127.0.0.1:5959 <host>`.

## What you get

| | |
|---|---|
| Provider | `commandcode-oauth` (aliases `commandcode-alpha`, `commandcode-account`) |
| Sign-in | `hermes auth add / status / logout commandcode-oauth` |
| Models | live account catalog in `hermes model` and `/model`; offline fallback list |
| Usage | `hermes usage` / `/usage`: 5-hour and weekly windows, credits left, billing period |
| Streaming | token streaming, reasoning deltas, tool calls |
| Images | flattened with `mimeType`; text-only models get a `[image: …]` placeholder instead of a silent drop |
| Prompt cache | `cacheReadTokens` reported as cached tokens |
| Errors | 402 → billing, 401 → auth, 403 → model entitlement, 429 → rate limit, … so failover works |
| Auxiliary tasks | vision, titles and compression run over the same transport (sync and async) |

## How it plugs in

Only documented `ProviderProfile` seams — no Hermes core file is modified:
`create_client`, `fetch_models`, `fetch_account_usage`, `auth_handler`, `refresh_credential`,
`classify_api_error`. The client declares `HERMES_SKIP_TRANSPORT_WRAP` and
`HERMES_SKIP_ASYNC_WRAP` so Hermes never rebuilds it as an OpenAI client against the wrong
endpoint.

| File | Responsibility |
|---|---|
| `wire.py` | Pure request/response translation (no Hermes imports) |
| `client.py` | HTTP + NDJSON transport, OpenAI-client shaped |
| `auth.py` | CLI-grant import, browser hand-off, pool hooks |
| `errors.py` | Failover classification |
| `__init__.py` | Profile registration, catalog, account usage |

## Status and caveats

* `/alpha/generate` is **not a documented public API**. It is what the official CLI uses and it
  can change without notice; this plugin tracks it. Hermes maintainers asked for this
  integration to live outside core for exactly that reason (NousResearch/hermes-agent#105967).
* Check Command Code's terms for your plan before using it from a third-party client.
* The sign-in token has no refresh grant. When it is revoked, `hermes auth add commandcode-oauth`
  again (or sign in with the CLI — the plugin picks the new grant up on the next 401).

## Development

```bash
cd tests && HERMES_TREE=/path/to/hermes-agent python -m pytest -q
```

Tests are hermetic: a local server plays `/alpha/generate`; no network or account needed.

## Credits

Builds on the review and protocol findings from NousResearch/hermes-agent#105968
(@Enough1122, @NanamiMio) and NanamiMio's
[hermes-commandcode-oauth](https://github.com/NanamiMio/hermes-commandcode-oauth).

MIT licensed.

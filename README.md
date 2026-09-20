# Eugene Plexus — `inference-driver`

[![CI](https://github.com/eugene-plexus/inference-driver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/inference-driver/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

A uniform HTTP surface over **one** model backend, for the [Eugene Plexus](https://github.com/eugene-plexus) control plane.

An install runs **N instances, one per backend**, with the [`gateway`](https://github.com/eugene-plexus/gateway) routing above them. A backend is any of:

- a local engine speaking OpenAI-compatible HTTP — llama.cpp's `llama-server`, vLLM, LM Studio, Ollama
- a hosted OpenAI-compatible API, such as OpenAI, xAI or OpenRouter
- a subprocess CLI riding a subscription you already pay for — `claude_code_cli`, `codex_cli`

That last kind is why this is a separate process rather than something folded into the gateway: a CLI has no HTTP endpoint to proxy to, so *something* has to front it, and normalising every backend through one surface is cheaper than special-casing the ones that can't be proxied. It's also what lets a driver run next to its engine on a remote GPU host while the gateway reaches it over the tailnet.

## What this service is — and what it isn't

A driver wraps one backend, serves `POST /v1/generate`, and persists its own config.
Its operator-supplied name lives in agent topology; the gateway learns what it
serves from `GET /v1/info`. A supervised driver can follow a runtime by name,
resolving the engine URL through its owning agent.

It decides **no** output-affecting parameter. Temperature, max tokens and stop
sequences arrive on the request or are not sent to the backend at all. The gateway
owns request values and its configured defaults; library launch profiles are a
separate concern. The driver never substitutes a local sampling default.

Correspondingly it **never refuses a model.** A backend that rejects `temperature`, as some reasoning models do, gets the unsupported parameter dropped and a warning logged — not a failed construction. You own the model; routing to it is the whole job.

## Status

**Pre-1.0, current through M7 (2026-09-10).** The HTTP surface, authenticated
config protocol, and `claude_code_cli`, `codex_cli`, and `openai_compat_http`
adapters are implemented. `runtimeName` follows a supervised runtime through
the agent and takes precedence over `baseUrl`. The agent now declares companion
drivers automatically. Local llama.cpp routing is live-verified; a real vLLM run
is still pending. Streaming (`/v1/generate/stream`) remains a **501 stub**.

## Wire contract

This service implements the [`inference-driver.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/inference-driver.yaml) OpenAPI 3.1 spec from the [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) repo. Pydantic models in `src/eugene_plexus_inference_driver/_generated/` are produced via codegen (see [Codegen](#codegen)).

Endpoints:

| Method | Path                  | Status     |
| ------ | --------------------- | ---------- |
| GET    | `/healthz`            | ✅          |
| GET    | `/v1/info`            | ✅          |
| GET    | `/v1/config`          | ✅          |
| GET    | `/v1/config/schema`   | ✅          |
| PATCH  | `/v1/config`          | ✅          |
| POST   | `/v1/config/test`     | ✅          |
| POST   | `/v1/generate`        | ✅          |
| POST   | `/v1/generate/stream` | stub (501) |

## Backends (adapters)

Three adapter classes share a provider registry. The driver's config picks a
`provider` (for example `claude_subscription`, `openai`, `minimax`, or
`ollama_local`); the registry maps it to an adapter and default base URL.

| Adapter              | Providers it serves                                                                                | Notes                                                                                                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `claude_code_cli`    | `claude_subscription`                                                                              | Wraps the `claude` CLI. Uses your Claude Pro/Max subscription — no API billing. System prompts are passed via `--system-prompt` so persona control is preserved.                                                                                                                          |
| `codex_cli`          | `chatgpt_subscription`                                                                             | Wraps `codex-cli`. Uses your ChatGPT subscription. ⚠️ Codex CLI has no `--system-prompt` equivalent; persona override and cwd-injection cannot be suppressed from the CLI surface. Use the `openai` provider via `openai_compat_http` if you need full persona control on the OpenAI side. |
| `openai_compat_http` | `openai`, `xai`, `openrouter`, `minimax`, `ollama_local`, `lmstudio_local`, `openai_compat_custom` | Direct HTTP to an OpenAI-compatible `/v1/chat/completions` endpoint. Custom providers accept `baseUrl` or follow a supervised `runtimeName`. Unsupported sampling parameters are adapted, not grounds for refusing the model.                                                             |
| `anthropic_api`      | Not implemented                                                                                    | Native Anthropic HTTP support is not one of the three implemented adapters.                                                                                                                                                                                                               |

The CLI adapters use an operator-installed, authenticated CLI. Subscription
availability and limits depend on the provider; this project does not supply
credentials or bypass provider restrictions.

## Running

### From source

```bash
pip install -e ".[dev]"
python -m eugene_plexus_inference_driver
```

By default it listens on `http://127.0.0.1:8081`, overridable via `EUGENE_PLEXUS_DRIVER_BIND_PORT` (the agent uses this when supervising). Configure runtime behavior via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

### Pairing with the gateway

For a supervised engine, create a runtime on the agent: it declares the companion
driver, configures `runtimeName`, and advertises the driver to peers. The gateway
discovers it from agent topology and only routes while its runtime is `ready`.

For a cloud or CLI backend, declare a separate `inference-driver` component on the
agent and configure its provider, model and credentials through the config trio.
There is no gateway `drivers` URL list. Replicas share a model id; fallback tiers
are configured through gateway `modelSlots`. Ports are per-instance assignments;
8082 is the library's default, not a reserved second-driver port.

### Configuration

Every config field is editable at runtime via `PATCH /v1/config`. `GET /v1/config/schema` returns UI-renderable metadata for every field — a generic UI renders the editor with no per-component code. The `eugene-plexus/ui` repo's config tab does this.

The driver also implements `POST /v1/config/test` — given an optional `overrides` body, it builds a temporary adapter from saved-merged-with-overrides config and runs a minimal `generate("Reply with PING")` round-trip. The UI's Test button on each driver's config tab uses this.

#### Degraded mode

If adapter construction fails at startup (missing API key, missing CLI binary, malformed config), the driver does **not** crash. It comes up in degraded mode with a working `/v1/config` and `/v1/config/schema` so the operator can fix the broken field via PATCH and restart. `/v1/generate` returns a 503 with a clear `Problem` message until the config is corrected.

#### Authentication

Under supervision, the agent supplies the install signing key and a service token.
`/v1/info` and generation accept authorized callers; config and restart require an
operator token. `/healthz` is unauthenticated. Standalone operation without auth
credentials is development-only and must not be exposed to an untrusted network.
Use a mesh VPN between hosts; transport isolation does not replace application auth.

## Codegen

Pydantic models for the wire contract are generated from the pinned commit of `eugene-plexus/specs` recorded in [`SPECS_REF`](SPECS_REF):

```bash
python scripts/codegen.py
```

The script downloads the specs at the pinned SHA, runs `datamodel-code-generator` against them, and writes Pydantic v2 models to `src/eugene_plexus_inference_driver/_generated/`. **The generated files are committed** so builds are reproducible without network access. CI re-runs codegen and fails the build if the working tree differs.

To bump to a newer specs commit:

```bash
echo "<new-sha>" > SPECS_REF
python scripts/codegen.py
```

…then commit `SPECS_REF` and the regenerated `_generated/` directory together.

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check .
ruff format --check .

# Type-check
mypy src/

# Test
pytest

# Codegen freshness
python scripts/codegen.py && git diff --exit-code src/eugene_plexus_inference_driver/_generated/
```

The test suite covers the HTTP surface, config and auth, runtime lookup, adapter
shape adaptation, degraded-mode startup, and subprocess encoding on Windows.
CLI live tests are gated behind `EUGENE_PLEXUS_DRIVER_LIVE_CLI=1` and API live
tests behind `EUGENE_PLEXUS_DRIVER_LIVE_API=1`.

## CLI environment

CLI subprocesses, including streaming calls, inherit backend credentials and proxy
settings but no `EUGENE_PLEXUS_*` variables. UTF-8 settings still apply. This prevents
accidental control-plane credential inheritance; it does not sandbox a CLI running
as the driver's OS user. In Ed25519 mode the driver receives only public verification
PEM through `EUGENE_PLEXUS_DRIVER_AUTH_VERIFY_KEY`, plus its service token and separate
master encryption key. Legacy HS256 bootstrap remains available until the install's
explicit key rotation; update every component before rotating. Private signing PEM
is never accepted by the driver.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

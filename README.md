# Eugene Plexus — `inference-driver`

[![CI](https://github.com/eugene-plexus/inference-driver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/inference-driver/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

A uniform HTTP surface over **one** model backend, for the [Eugene Plexus](https://github.com/eugene-plexus) control plane.

An install runs **N instances, one per backend**, with the [`gateway`](https://github.com/eugene-plexus/gateway) routing above them. A backend is any of:

- a local engine speaking OpenAI-compatible HTTP — llama.cpp's `llama-server`, vLLM, LM Studio, Ollama
- a hosted API — Anthropic, OpenAI, xAI, OpenRouter
- a subprocess CLI riding a subscription you already pay for — `claude_code_cli`, `codex_cli`

That last kind is why this is a separate process rather than something folded into the gateway: a CLI has no HTTP endpoint to proxy to, so *something* has to front it, and normalising every backend through one surface is cheaper than special-casing the ones that can't be proxied. It's also what lets a driver run next to its engine on a remote GPU host while the gateway reaches it over the tailnet.

## What this service is — and what it isn't

A driver is **anonymous and stateless**. It wraps one backend, serves `POST /v1/generate`, and does not know its position in any topology. Its operator-supplied name lives in the watchdog topology; the gateway learns what this driver *serves* by reading `GET /v1/info` and routes on that.

It also decides **no** output-affecting parameter. Temperature, max tokens and stop sequences arrive on the request or are not sent to the backend at all — the gateway owns them and resolves them from the model's settings profile. The driver applies what it's given and never substitutes a local default.

Correspondingly it **never refuses a model.** A backend that rejects `temperature`, as some reasoning models do, gets the unsupported parameter dropped and a warning logged — not a failed construction. You own the model; routing to it is the whole job.

## Status

**v0.1, working.** The HTTP surface, config protocol with `/v1/config/test`, and three adapters (`claude_code_cli`, `codex_cli`, `openai_api`) are wired up end-to-end. Streaming (`/v1/generate/stream`) is still a 501 stub — it lands alongside the gateway + UI consumers.

## Wire contract

This service implements the [`inference-driver.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/inference-driver.yaml) OpenAPI 3.1 spec from the [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) repo. Pydantic models in `src/eugene_plexus_inference_driver/_generated/` are produced via codegen (see [Codegen](#codegen)).

Endpoints:

| Method | Path                  | Status     |
|--------|-----------------------|------------|
| GET    | `/healthz`            | ✅          |
| GET    | `/v1/info`            | ✅          |
| GET    | `/v1/config`          | ✅          |
| GET    | `/v1/config/schema`   | ✅          |
| PATCH  | `/v1/config`          | ✅          |
| POST   | `/v1/config/test`     | ✅          |
| POST   | `/v1/generate`        | ✅          |
| POST   | `/v1/generate/stream` | stub (501) |

## Backends (adapters)

v0.1 ships with three adapter classes plus a provider registry. The gateway-facing config picks a `provider` (e.g. `claude_subscription`, `openai`, `minimax`, `ollama_local`); the registry maps each provider to the right adapter with the right base URL.

| Adapter                | Providers it serves | Notes |
|------------------------|---------------------|-------|
| `claude_code_cli`      | `claude_subscription` | Wraps the `claude` CLI. Uses your Claude Pro/Max subscription — no API billing. System prompts are passed via `--system-prompt` so persona control is preserved. |
| `codex_cli`            | `chatgpt_subscription` | Wraps `codex-cli`. Uses your ChatGPT subscription. ⚠️ Codex CLI has no `--system-prompt` equivalent; persona override and cwd-injection cannot be suppressed from the CLI surface. Use the `openai` provider via `openai_compat_http` if you need full persona control on the OpenAI side. |
| `openai_compat_http`   | `openai`, `xai`, `openrouter`, `minimax`, `ollama_local`, `lmstudio_local`, `openai_compat_custom` | Direct HTTP to any OpenAI-compatible `/v1/chat/completions` endpoint. Provider registry sets the base URL and any deny patterns (e.g. OpenAI's reasoning-only models that reject `temperature`); the `openai_compat_custom` provider lets the operator point at any URL with their own `baseUrl`. Pay-per-token when the provider charges; free for local backends. |
| `anthropic_api`        | (planned for v0.2+) | Direct HTTP to Anthropic. Pay-per-token. |

The CLI adapters are **primary production mode for personal installations** — they run on the AI subscription you already pay for, no separate API bill.

## Running

### From source

```bash
pip install -e ".[dev]"
python -m eugene_plexus_inference_driver
```

By default it listens on `http://127.0.0.1:8081`, overridable via `EUGENE_PLEXUS_DRIVER_BIND_PORT` (the watchdog uses this when supervising). Configure runtime behavior via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

### Pairing with the gateway

Run two driver instances on different ports — typically 8081 and 8082 — each with a different `adapter` config. Then point the gateway's `drivers` config at both:

```yaml
drivers:
  - name: left
    url: http://127.0.0.1:8081
  - name: right
    url: http://127.0.0.1:8082
```

The gateway's UI exposes a per-driver Test button that calls each driver's `/v1/info` so you can verify the URLs are reachable before saving.

### Configuration

Every config field is editable at runtime via `PATCH /v1/config`. `GET /v1/config/schema` returns UI-renderable metadata for every field — a generic UI renders the editor with no per-component code. The `eugene-plexus/ui` repo's config tab does this.

The driver also implements `POST /v1/config/test` — given an optional `overrides` body, it builds a temporary adapter from saved-merged-with-overrides config and runs a minimal `generate("Reply with PING")` round-trip. The UI's Test button on each driver's config tab uses this.

#### Degraded mode

If adapter construction fails at startup (missing API key, missing CLI binary, malformed config), the driver does **not** crash. It comes up in degraded mode with a working `/v1/config` and `/v1/config/schema` so the operator can fix the broken field via PATCH and restart. `/v1/generate` returns a 503 with a clear `Problem` message until the config is corrected.

#### v0.1 auth

> **v0.1 has no application auth.** Deployment assumption: behind a [Tailscale](https://tailscale.com/) tailnet or equivalent network boundary. Anyone reachable on the network can read and modify config — including secrets like API keys via PATCH. Auth lands in v0.2.

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

The test suite (35 tests, ~1s) covers the HTTP surface, the config protocol, all three adapters' shape adaptation, degraded-mode startup, and a UTF-8 round-trip pinning the subprocess encoding fix on Windows. CLI live-fire tests are gated behind `EUGENE_PLEXUS_DRIVER_LIVE_CLI=1` and the API live-fire test behind `EUGENE_PLEXUS_DRIVER_LIVE_API=1`.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

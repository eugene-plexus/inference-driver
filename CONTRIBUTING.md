# Contributing to Eugene Plexus `inference-driver`

Thanks for your interest. This service implements the `inference-driver` OpenAPI contract from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) — please read this before opening a PR.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a CLA. **Every commit must be signed off** with `git commit -s`:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your `git config user.name` and `git config user.email`. CI blocks PRs whose commits are missing matching sign-offs.

If you forgot to sign off, fix the most recent commit:

```bash
git commit --amend -s --no-edit
```

…or for a whole branch:

```bash
git rebase --signoff main
```

The full DCO text is in [the specs CONTRIBUTING.md](https://github.com/eugene-plexus/specs/blob/main/CONTRIBUTING.md).

## Wire contract changes go in `specs`, not here

If your change touches the HTTP API — endpoints, request/response shapes, schemas — it belongs in [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs), not here. Land that PR first; bump `SPECS_REF` and re-run codegen here in a follow-up.

PRs to this repo should generally cover one or more of:

- **Implementation** of an existing spec endpoint (e.g. wiring up a stub).
- **Adapter work** — new backend adapter, fixes to an existing one, refactors to the adapter base class.
- **Reliability / performance** — better error handling, streaming improvements, etc.
- **Tooling** — CI, type-checking, lint config, codegen script.

## Local setup

```bash
git clone https://github.com/eugene-plexus/inference-driver
cd inference-driver
python -m venv .venv
. .venv/bin/activate           # or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Git hooks

We use [pre-commit](https://pre-commit.com/) to auto-format staged files with Ruff before they reach CI. Enable it once per clone:

```bash
pip install pre-commit
pre-commit install
```

After that, `git commit` runs `ruff check --fix` and `ruff format` on staged Python files; if a hook reformats anything, re-stage and commit again.

## Style

- **Python 3.12+** features are fine. We use `from __future__ import annotations` only where it materially helps; otherwise just use modern syntax.
- **Ruff** for lint and format. `ruff check .` and `ruff format .` should both be clean before you push. CI enforces.
- **Mypy strict** for type-checking. New code must type-check; the `_generated/` directory is excluded.
- **No comments explaining what code does** — let names do the work. Reserve comments for *why* a non-obvious choice was made.
- **Async-first.** Every adapter and route handler is `async`. Don't introduce synchronous I/O on the request path.

## Running checks

```bash
ruff check .                       # lint
ruff format --check .              # formatting
mypy src/                          # type-check
pytest                             # tests
python scripts/codegen.py          # regenerate models from pinned specs
git diff --exit-code src/.../_generated/   # codegen freshness
```

## Reporting issues

File issues at <https://github.com/eugene-plexus/inference-driver/issues>. Useful issues include:

- Concrete adapter failures with reproduction steps.
- Spec-vs-impl divergence (the impl drifted from `inference-driver.yaml`).
- Performance / latency regressions with measurements.

For broader architectural questions about Eugene Plexus, file the issue on the [gateway repo](https://github.com/eugene-plexus/gateway) instead.

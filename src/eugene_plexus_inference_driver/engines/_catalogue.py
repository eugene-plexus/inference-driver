"""A provider account's model list: read, filtered, kept, and refreshed.

**P1 (2026-09-27): one driver per provider account.** An OpenAI-compatible
driver with no `modelId` configured serves every model its backend lists --
all of OpenRouter, every model an Ollama has pulled, whatever a
`llama-server` in router mode holds. This module is that list.

Four rules, each from a mistake this project already made once:

* **A failed read never empties the list** (R2.1). The previous good list
  stays in force and the reason is reported on `/v1/info`. An upstream that
  blips for one refresh must not unroute six hundred models.
* **The last good list is kept on disk** beside the driver's config, so a
  driver restarted while its upstream is down still serves what it served.
  It is thrown away when the provider or address it came from changes: a
  list from another backend is not this backend's.
* **The listing is the provider's own, not a heuristic** (call P1-3).
  OpenRouter's catalogue says per model what it takes, gives back and
  accepts; Ollama's `/api/show` names each model's capabilities; LM
  Studio's `/api/v0/models` names its type. Only where a provider's list
  says nothing (OpenAI's `/v1/models`) does a model inherit the answer the
  driver gives for itself.
* **Filters are read live.** `catalogueInclude` and `catalogueExclude` take
  effect on the next `/v1/info`, no restart, because narrowing what an
  account exposes should not cost every in-flight request.

Measured 2026-09-27 (`docs/acceptance/provider-accounts-measurement.md`):
OpenRouter's `/models` is the same 628 with or without a key, while
`/models/user?output_modalities=all` is the account's own 625 -- it drops
the models this account's settings cannot call. The default `/models` also
hides the 170 whose output is not text. So the account listing is the one.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .._generated.models import Capabilities, DriverCatalogue, DriverModel
from .._private_files import write_private_text
from ..speech import ALL_FORMATS, OPENROUTER_FORMATS

log = logging.getLogger(__name__)

#: Default seconds between catalogue reads. An hour: OpenRouter's list moves
#: daily, and a model the operator just pulled into Ollama is one restart
#: (or one hour) away. Configurable per driver.
DEFAULT_REFRESH_MINUTES = 60

#: The first retry after a failed read, doubled per failure up to the normal
#: interval. A key being rotated or an upstream blipping should not wait an
#: hour to be noticed as fixed.
_FIRST_RETRY_SECONDS = 30.0

#: Per-request deadline for a listing. Generous for a 625-entry body, short
#: enough that a hung upstream does not hold a refresh for minutes.
_LIST_TIMEOUT = httpx.Timeout(20.0, connect=5.0)

#: Ollama needs one `/api/show` per model; bounded so twenty models are
#: twenty quick local calls, not a burst.
_SHOW_CONCURRENCY = 4
_SHOW_TIMEOUT = httpx.Timeout(5.0, connect=2.0)

#: OpenRouter's `supported_parameters` names, mapped to the request-setting
#: names A2's `callerSettings` uses. Only settings the OpenAI-compatible
#: engine can carry at all appear; the rest of OpenRouter's list (logprobs,
#: reasoning, verbosity, ...) waits for P2's fields.
_OPENROUTER_SETTINGS: tuple[tuple[str, str], ...] = (
    ("max_tokens", "maxTokens"),
    ("temperature", "temperature"),
    ("top_p", "topP"),
    ("seed", "seed"),
    ("stop", "stop"),
    ("tools", "tools"),
    ("tool_choice", "toolChoice"),
    ("response_format", "responseFormat"),
    ("top_k", "topK"),
    ("min_p", "minP"),
    ("frequency_penalty", "frequencyPenalty"),
    ("presence_penalty", "presencePenalty"),
    ("parallel_tool_calls", "parallelToolCalls"),
    # P2c (2026-09-28). `top_logprobs` rides with `logprobs`, which every
    # model listing one lists both (149 each, measured).
    ("logprobs", "logprobs"),
    ("logit_bias", "logitBias"),
    ("reasoning_effort", "reasoningEffort"),
    ("verbosity", "verbosity"),
    ("prediction", "prediction"),
    ("web_search_options", "webSearchOptions"),
)


class CatalogueError(Exception):
    """A catalogue read that produced no list, with the reason in words."""


def matches(pattern: str, value: str) -> bool:
    """`*` matches any run of characters, `/` included; nothing else is special.

    Deliberately not `fnmatch`: `[`, `?` and `\\` are ordinary characters in
    a model id (`~`, `:` and `/` already are), and one wildcard is a rule an
    operator can hold in their head. The same function is copied into the
    gateway, the agent and the control root for `allowedModels`.
    """
    if "*" not in pattern:
        return pattern == value
    parts = [re.escape(part) for part in pattern.split("*")]
    return re.fullmatch(".*".join(parts), value, flags=re.DOTALL) is not None


def exposed_by(include: list[str], exclude: list[str], model_id: str) -> bool:
    """Kept when some include pattern matches and no exclude pattern does."""
    if not any(matches(p, model_id) for p in include):
        return False
    return not any(matches(p, model_id) for p in exclude)


@dataclass(frozen=True)
class EngineDefaults:
    """What a model inherits where its provider's listing says nothing (P1-3):
    the answer the driver gives for itself."""

    supported_settings: list[str]
    tool_calling: bool
    streaming: bool


# ---------------------------------------------------------------------------
# Per-source mapping. Each takes the upstream's own body and returns the
# models it describes, never raising on one malformed entry: a single odd
# row in a 625-row list must not cost the other 624.
# ---------------------------------------------------------------------------


def surfaces_from_output(output: list[str], inputs: list[str]) -> list[str]:
    """What requests a model answers, from OpenRouter's output modalities.

    `text` output is chat, whatever else comes with it -- Lyria's music and
    Gemini's images both ride chat completions (measured 2026-09-27). An
    image-only model is `image`. The rest are named after the modality.
    """
    out: list[str] = []
    if "text" in output:
        out.append("chat")
    for modality, surface in (
        ("embeddings", "embeddings"),
        ("decisions", "decisions"),
        ("speech", "speech"),
        ("transcription", "transcription"),
        ("rerank", "rerank"),
        ("video", "video"),
    ):
        if modality in output:
            out.append(surface)
    if "image" in output and "text" not in output:
        out.append("image")
    return out


def from_openrouter(body: Any) -> list[DriverModel]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise CatalogueError("OpenRouter's model list had no `data` array")
    models: list[DriverModel] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        raw_arch = entry.get("architecture")
        arch: dict[str, Any] = raw_arch if isinstance(raw_arch, dict) else {}
        inputs = [m for m in (arch.get("input_modalities") or []) if isinstance(m, str)]
        output = [m for m in (arch.get("output_modalities") or []) if isinstance(m, str)]
        params = {p for p in (entry.get("supported_parameters") or []) if isinstance(p, str)}
        settings = [ours for theirs, ours in _OPENROUTER_SETTINGS if theirs in params]
        if "structured_outputs" in params and "responseFormat" not in settings:
            settings.append("responseFormat")
        surfaces = surfaces_from_output(output, inputs)
        context = entry.get("context_length")
        voices = entry.get("supported_voices")
        models.append(
            DriverModel(
                id=entry["id"],
                name=entry.get("name") if isinstance(entry.get("name"), str) else None,
                surfaces=surfaces,
                inputModalities=inputs or None,
                outputModalities=output or None,
                voices=[v for v in voices if isinstance(v, str)]
                if isinstance(voices, list)
                else None,
                capabilities=Capabilities(
                    supportedSettings=settings,
                    streaming="chat" in surfaces,
                    imageInput="image" in inputs,
                    audioInput="audio" in inputs,
                    fileInput="file" in inputs,
                    # From what the model gives back, not from its
                    # parameter list: no audio-output model on OpenRouter
                    # lists `modalities` or `audio` there (P2b, measured).
                    audioOutput="audio" in output,
                    # P3a: OpenRouter's speech route makes mp3 and pcm, and
                    # the driver makes WAV from pcm (measured 2026-09-28).
                    speechFormats=list(OPENROUTER_FORMATS) if "speech" in surfaces else None,
                    toolCalling="tools" in params,
                    maxContextTokens=context if isinstance(context, int) and context > 0 else None,
                ),
            )
        )
    return models


def ollama_entry(name: str, show: dict[str, Any] | None, defaults: EngineDefaults) -> DriverModel:
    """One pulled model. `capabilities` is on `/api/show` from Ollama 0.6;
    an older Ollama, or a failed show, inherits the driver's own answer."""
    caps = show.get("capabilities") if isinstance(show, dict) else None
    if isinstance(caps, list):
        names = {c for c in caps if isinstance(c, str)}
        surfaces: list[str] = []
        if "completion" in names:
            surfaces.append("chat")
        if "embedding" in names:
            surfaces.append("embeddings")
        return DriverModel(
            id=name,
            surfaces=surfaces,
            inputModalities=["text", "image"] if "vision" in names else ["text"],
            capabilities=Capabilities(
                supportedSettings=list(defaults.supported_settings) if "chat" in surfaces else [],
                streaming=defaults.streaming and "chat" in surfaces,
                imageInput="vision" in names,
                toolCalling="tools" in names,
                # `/api/show` carries only the trained maximum, which
                # overstates whatever Ollama resolved (step 7 measured
                # 131072 auto-sized against a trained number above it).
                # Unknown is the honest answer here.
                maxContextTokens=None,
            ),
        )
    return _inherited(name, defaults)


def _inherited(
    model_id: str, defaults: EngineDefaults, *, surfaces: list[str] | None = None
) -> DriverModel:
    served = ["chat"] if surfaces is None else surfaces
    return DriverModel(
        id=model_id,
        surfaces=served,
        capabilities=Capabilities(
            supportedSettings=list(defaults.supported_settings) if "chat" in served else [],
            streaming=defaults.streaming and "chat" in served,
            # Never assumed: attachments route only where a listing said so.
            imageInput=False,
            audioInput=False,
            fileInput=False,
            audioOutput=False,
            toolCalling=defaults.tool_calling and "chat" in served,
            maxContextTokens=None,
        ),
    )


def from_lmstudio(body: Any, defaults: EngineDefaults) -> list[DriverModel]:
    """LM Studio's `/api/v0/models`: `type` is `llm`, `vlm` or `embeddings`.

    **Written from LM Studio's documentation, not measured** -- no LM Studio
    was available on 2026-09-27. A `capabilities` array naming `tool_use`
    is read when present; otherwise tool calling is inherited.
    """
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise CatalogueError("LM Studio's model list had no `data` array")
    models: list[DriverModel] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        kind = entry.get("type")
        if kind == "embeddings":
            models.append(_inherited(entry["id"], defaults, surfaces=["embeddings"]))
            continue
        model = _inherited(entry["id"], defaults)
        caps = entry.get("capabilities")
        assert model.capabilities is not None
        if isinstance(caps, list):
            model.capabilities.toolCalling = "tool_use" in caps
        if kind == "vlm":
            model.capabilities.imageInput = True
            model.inputModalities = ["text", "image"]
        loaded = entry.get("loaded_context_length")
        if isinstance(loaded, int) and loaded > 0:
            model.capabilities.maxContextTokens = loaded
        models.append(model)
    return models


def from_openai_list(
    body: Any, defaults: EngineDefaults, *, classify: Callable[[str], list[str]] | None
) -> list[DriverModel]:
    """The OpenAI `/v1/models` shape, which says nothing per model.

    `classify` sorts OpenAI's own list by id (embeddings, speech, images...),
    since `api.openai.com` returns every model on the account; for a local
    or custom server every model is chat (the operator chose what is on it).
    vLLM's `max_model_len` is the window it resolved and is taken when
    present.
    """
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise CatalogueError("the model list had no `data` array")
    models: list[DriverModel] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        surfaces = classify(entry["id"]) if classify is not None else None
        model = _inherited(entry["id"], defaults, surfaces=surfaces)
        if "speech" in model.surfaces and model.capabilities is not None:
            # Only OpenAI's own list sorts a model into speech, and its API
            # makes all six formats (P3a).
            model.capabilities.speechFormats = list(ALL_FORMATS)
        window = entry.get("max_model_len")
        if isinstance(window, int) and window > 0 and model.capabilities is not None:
            model.capabilities.maxContextTokens = window
        models.append(model)
    return models


def upstream_words(response: httpx.Response) -> str:
    """The upstream's own reason, where it gave one, for `catalogue.error`.

    A scoped key refused a listing is not an invalid key (ElevenLabs:
    *"missing the permission models_read"*, measured), so the words matter
    more than the status.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip() or f"HTTP {response.status_code}"
    paths = (("error", "message"), ("detail", "message"), ("detail",), ("message",), ("error",))
    for path in paths:
        value: Any = body
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()[:300]
    return f"HTTP {response.status_code}"


# ---------------------------------------------------------------------------
# The catalogue itself.
# ---------------------------------------------------------------------------


class Catalogue:
    """One account's model list, as last read and as currently filtered."""

    def __init__(
        self,
        *,
        source: str,
        origin: str,
        fetch: Callable[[], Awaitable[list[DriverModel]]],
        get: Callable[[str], Any],
        path: Path | None = None,
    ) -> None:
        self.source = source
        #: Where the list came from -- the provider plus the address. A
        #: persisted list from any other origin is discarded on load.
        self._origin = origin
        self._fetch = fetch
        self._get = get
        self._path = path
        self._models: list[DriverModel] = []
        self._by_id: dict[str, DriverModel] = {}
        self._refreshed_at: _dt.datetime | None = None
        self._error: str | None = None
        self._lock = asyncio.Lock()
        self._load()

    # -- filters -----------------------------------------------------------

    def patterns(self) -> tuple[list[str], list[str]]:
        """`(include, exclude)`, read live. An unset include is `["*"]`
        (call #2: an aggregator exposes everything); an include the operator
        emptied on purpose exposes nothing, which is what it says."""
        include = self._get("catalogueInclude")
        exclude = self._get("catalogueExclude")
        if isinstance(include, list):
            inc = [p for p in include if isinstance(p, str) and p]
        else:
            inc = ["*"]
        exc = [p for p in exclude if isinstance(p, str) and p] if isinstance(exclude, list) else []
        return inc, exc

    def refresh_seconds(self) -> int:
        raw = self._get("catalogueRefreshMinutes")
        minutes = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else None
        return 60 * (minutes or DEFAULT_REFRESH_MINUTES)

    def exposed(self) -> list[DriverModel]:
        include, exclude = self.patterns()
        return [m for m in self._models if exposed_by(include, exclude, m.id)]

    def find(self, model_id: str) -> DriverModel | None:
        """The model, if this account lists it AND its patterns keep it."""
        model = self._by_id.get(model_id)
        if model is None:
            return None
        include, exclude = self.patterns()
        return model if exposed_by(include, exclude, model.id) else None

    def summary(self) -> DriverCatalogue:
        include, exclude = self.patterns()
        return DriverCatalogue(
            source=self.source,
            total=len(self._models),
            exposed=len(self.exposed()),
            include=include,
            exclude=exclude,
            refreshedAt=self._refreshed_at,
            refreshSeconds=self.refresh_seconds(),
            error=self._error,
        )

    # -- reading -----------------------------------------------------------

    async def refresh(self) -> bool:
        """Read the list once. True on success; on failure the previous list
        stays and `error` says why."""
        async with self._lock:
            try:
                models = await self._fetch()
            except CatalogueError as e:
                self._error = str(e)
            except httpx.HTTPError as e:
                self._error = f"could not reach the model list: {e!r}"
            except Exception as e:  # never let a refresh kill the loop
                log.exception("catalogue read failed unexpectedly")
                self._error = f"the model list could not be read: {e}"
            else:
                self._set(models, _dt.datetime.now(_dt.UTC))
                self._error = None
                self._save()
                log.info(
                    "catalogue (%s): %d models listed, %d exposed",
                    self.source,
                    len(self._models),
                    len(self.exposed()),
                )
                return True
            log.warning(
                "catalogue (%s) read failed; keeping the %d models already known: %s",
                self.source,
                len(self._models),
                self._error,
            )
            return False

    async def run(self) -> None:
        """Read now, then on the interval, backing off after a failure.
        Cancelled with the driver."""
        retry = _FIRST_RETRY_SECONDS
        while True:
            ok = await self.refresh()
            interval = float(self.refresh_seconds())
            if ok:
                retry = _FIRST_RETRY_SECONDS
                delay = interval
            else:
                delay = min(retry, interval)
                retry = min(retry * 2, interval)
            await asyncio.sleep(delay)

    def _set(self, models: list[DriverModel], when: _dt.datetime | None) -> None:
        seen: dict[str, DriverModel] = {}
        for model in models:
            seen.setdefault(model.id, model)
        self._models = sorted(seen.values(), key=lambda m: m.id)
        self._by_id = {m.id: m for m in self._models}
        self._refreshed_at = when

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if raw.get("origin") != self._origin:
                log.info("ignoring %s: it is another backend's model list", self._path)
                return
            models = [DriverModel.model_validate(m) for m in raw.get("models") or []]
            when = raw.get("refreshedAt")
            self._set(models, _dt.datetime.fromisoformat(when) if isinstance(when, str) else None)
            log.info("catalogue (%s): %d models from the last good read", self.source, len(models))
        except (OSError, ValueError, TypeError, AttributeError) as e:
            # A damaged copy is a cache miss, never a driver that will not start.
            log.warning("could not read the saved model list %s: %s", self._path, e)

    def _save(self) -> None:
        if self._path is None:
            return
        body = {
            "origin": self._origin,
            "source": self.source,
            "refreshedAt": self._refreshed_at.isoformat() if self._refreshed_at else None,
            "models": [m.model_dump(mode="json", exclude_none=True) for m in self._models],
        }
        try:
            write_private_text(self._path, json.dumps(body))
        except OSError as e:
            log.warning("could not save the model list to %s: %s", self._path, e)

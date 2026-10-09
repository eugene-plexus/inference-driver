"""Gemini's wire, both ways: pure functions, no network (design gemini-provider.md).

Everything the `gemini_api` engine translates lives here so it can be tested
without a transport: the listing's models into the catalogue (with our own
table of what each family takes, because Google's listing names no input
modalities), OpenAI-shaped messages into `contents`, a `generateContent`
answer into the driver's result, Google's error JSON into words, and the
bounded memory of thought signatures (G2, G7).

Written from Google's documentation of 2026-10-09; what could not be
confirmed there is listed in the build report and checked by the live run.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import struct
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx

from .._generated.models import (
    Capabilities,
    DriverModel,
    FinishReason,
    FunctionCall,
    ImageCapabilities,
    ReasoningEffort,
    Role,
    SpeechFormat,
    ToolCall,
    Usage,
    VideoCapabilities,
)
from ..failures import retry_after
from ..images import content_wire
from ._subprocess import CliError

# ---------------------------------------------------------------------------
# What a model is: our table, over Google's listing
# ---------------------------------------------------------------------------

#: Gemini's prebuilt voices (speech-generation docs, 2026-10-09).
VOICES: tuple[str, ...] = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe",
    "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
    "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux",
    "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager",
    "Sulafat",
)  # fmt: skip

#: Gemini's speech output: PCM, 24 kHz, 16-bit, mono (G5).
SPEECH_FORMATS: tuple[SpeechFormat, ...] = (SpeechFormat.wav, SpeechFormat.pcm)

#: What a chat setting is worth on the wire; the rest are refused (A2).
CHAT_SETTINGS: tuple[str, ...] = (
    "maxTokens",
    "temperature",
    "topP",
    "seed",
    "stop",
    "tools",
    "toolChoice",
    "responseFormat",
    "topK",
    "frequencyPenalty",
    "presencePenalty",
    "parallelToolCalls",
)

#: Veo sizes the driver's `size` ("WxH") maps to `(aspectRatio, resolution)`.
VIDEO_SIZES: dict[str, tuple[str, str]] = {
    "1280x720": ("16:9", "720p"),
    "720x1280": ("9:16", "720p"),
    "1920x1080": ("16:9", "1080p"),
    "1080x1920": ("9:16", "1080p"),
    "3840x2160": ("16:9", "4k"),
    "2160x3840": ("9:16", "4k"),
}
VIDEO_SECONDS = (4, 6, 8)

_GENERATION = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


def bare_id(name: str) -> str:
    """`models/gemini-3.5-flash` -> `gemini-3.5-flash`."""
    return name.removeprefix("models/")


def generation_of(model_id: str) -> tuple[int, int] | None:
    """`(3, 5)` for `gemini-3.5-flash`; None for an id outside the Gemini line."""
    match = _GENERATION.match(model_id.lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def surfaces_of(model_id: str, methods: set[str]) -> list[str]:
    """What requests a model answers, from Google's methods and our family table."""
    lowered = model_id.lower()
    if methods & {"embedContent", "batchEmbedContents"} or "embedding" in lowered:
        return ["embeddings"]
    if "predictLongRunning" in methods or lowered.startswith("veo-"):
        return ["video"]
    if not methods & {"generateContent", "streamGenerateContent"}:
        return []
    if "-tts" in lowered:
        return ["speech"]
    if "transcribe" in lowered:
        return ["transcription"]
    if "-image" in lowered or lowered.endswith("-image"):
        return ["image"]
    if lowered.startswith(("gemini-live", "gemini-robotics")) or "native-audio" in lowered:
        return []
    # A Gemini chat model hears audio as well as reading text: it is a
    # transcriber too (the instruction is ours, the model's own).
    return ["chat", "transcription"] if lowered.startswith("gemini") else ["chat"]


def _chat_capabilities(entry: dict[str, Any], model_id: str) -> Capabilities:
    gemma = model_id.lower().startswith("gemma")
    thinking = entry.get("thinking") is True
    settings = [s for s in CHAT_SETTINGS if not (gemma and s in ("tools", "toolChoice"))]
    if thinking:
        settings.append("reasoningEffort")
    window = entry.get("inputTokenLimit")
    return Capabilities(
        supportedSettings=settings,
        streaming=True,
        # Gemini chat models take images, audio and PDFs inline; Gemma does not.
        imageInput=not gemma,
        audioInput=not gemma,
        fileInput=not gemma,
        toolCalling=not gemma,
        maxContextTokens=window if isinstance(window, int) and window > 0 else None,
    )


def model_from_listing(entry: dict[str, Any]) -> DriverModel | None:
    """One `models[]` entry, or None when it serves nothing Eugene routes."""
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return None
    model_id = bare_id(name)
    raw_methods = entry.get("supportedGenerationMethods")
    methods = (
        {m for m in raw_methods if isinstance(m, str)} if isinstance(raw_methods, list) else set()
    )
    surfaces = surfaces_of(model_id, methods)
    if not surfaces:
        return None
    display = entry.get("displayName")
    window = entry.get("inputTokenLimit")
    caps: Capabilities
    inputs: list[str]
    outputs: list[str]
    voices: list[str] | None = None
    if "chat" in surfaces:
        caps = _chat_capabilities(entry, model_id)
        gemma = model_id.lower().startswith("gemma")
        inputs = ["text"] if gemma else ["text", "image", "audio", "video", "file"]
        outputs = ["text"]
    elif "image" in surfaces:
        caps = Capabilities(
            supportedSettings=[],
            streaming=False,
            imageInput=True,
            maxContextTokens=window if isinstance(window, int) and window > 0 else None,
            image=ImageCapabilities(
                streaming=False,
                maxImages=1,
                minReferences=0,
                maxReferences=14,
                mask=False,
                qualities=[],
                backgrounds=[],
                outputFormats=None,
            ),
        )
        inputs, outputs = ["text", "image"], ["text", "image"]
    elif "video" in surfaces:
        lite = "lite" in model_id.lower()
        sizes = [s for s, (_, res) in VIDEO_SIZES.items() if not (lite and res == "4k")]
        caps = Capabilities(
            supportedSettings=[],
            video=VideoCapabilities(
                durations=list(VIDEO_SECONDS), sizes=sizes, firstFrame=True, prices=None
            ),
        )
        inputs, outputs = ["text", "image"], ["video"]
    elif "speech" in surfaces:
        caps = Capabilities(
            supportedSettings=[], speechFormats=list(SPEECH_FORMATS), audioOutput=False
        )
        inputs, outputs, voices = ["text"], ["audio"], list(VOICES)
    elif "transcription" in surfaces:
        caps = Capabilities(supportedSettings=[], audioInput=True)
        inputs, outputs = ["audio"], ["text"]
    else:
        caps = Capabilities(
            supportedSettings=[],
            maxContextTokens=window if isinstance(window, int) and window > 0 else None,
        )
        inputs, outputs = ["text"], []
    return DriverModel(
        id=model_id,
        name=display if isinstance(display, str) and display else None,
        surfaces=surfaces,
        inputModalities=inputs or None,
        outputModalities=outputs or None,
        voices=voices,
        capabilities=caps,
    )


# ---------------------------------------------------------------------------
# Thought signatures (G2, G7)
# ---------------------------------------------------------------------------

SIGNATURE_ENTRIES = 4096
SIGNATURE_SECONDS = 24 * 60 * 60.0


@dataclass(frozen=True)
class RememberedCall:
    name: str
    signature: str | None
    #: Whether Google itself gave the call an id (so it is sent back as one).
    google_id: bool


class SignatureCache:
    """Bounded memory of the calls a model made: 4,096 for 24 hours (G7).

    A Gemini 3 thinking model puts an encrypted `thoughtSignature` on the
    `functionCall` part it returns, and refuses the next turn of the
    conversation unless it comes back with the call. The harness on the
    other side is an OpenAI client that cannot carry the field, so the
    driver remembers it by tool-call id. In memory only: a restart loses
    it, and the one turn that needed it is refused by Google, which the
    engine then explains (`MISSING_SIGNATURE_WORDS`).
    """

    def __init__(
        self, *, entries: int = SIGNATURE_ENTRIES, seconds: float = SIGNATURE_SECONDS
    ) -> None:
        self._entries = entries
        self._seconds = seconds
        self._items: OrderedDict[str, tuple[float, RememberedCall]] = OrderedDict()

    def remember(self, call_id: str, call: RememberedCall) -> None:
        self._items.pop(call_id, None)
        self._items[call_id] = (time.perf_counter() + self._seconds, call)
        while len(self._items) > self._entries:
            self._items.popitem(last=False)

    def recall(self, call_id: str) -> RememberedCall | None:
        found = self._items.get(call_id)
        if found is None:
            return None
        expires, call = found
        if time.perf_counter() >= expires:
            del self._items[call_id]
            return None
        return call

    def __len__(self) -> int:
        return len(self._items)


MISSING_SIGNATURE_WORDS = (
    "This conversation's tool call was made before the driver restarted (or more than "
    "24 hours ago); Gemini needs its thought signature, which this driver kept only in "
    "memory, and refused the turn without it. Start the conversation again, or send the "
    "tool result to a model that does not need signatures."
)
_SIGNATURE_REFUSAL = re.compile(r"thought[ _]?signature", re.I)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _refusal(message: str) -> CliError:
    return CliError(message, upstream_status=400)


def _data_url(url: str, field: str) -> tuple[str, str]:
    header, separator, data = url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise _refusal(f"{field}: only inline base64 data URLs are sent to Gemini")
    return header[5:].removesuffix(";base64"), data


def _parts_of(content: Any, field: str) -> list[dict[str, Any]]:
    """An OpenAI message's content as Gemini parts."""
    wire = content_wire(content)
    if wire is None or wire == "":
        return []
    if isinstance(wire, str):
        return [{"text": wire}]
    parts: list[dict[str, Any]] = []
    for index, part in enumerate(wire):
        kind = part.get("type")
        where = f"{field}[{index}]"
        if kind == "text":
            parts.append({"text": part.get("text", "")})
        elif kind == "image_url":
            mime, data = _data_url(part["image_url"]["url"], where)
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        elif kind == "input_audio":
            audio = part["input_audio"]
            mime = "audio/wav" if audio.get("format") == "wav" else "audio/mp3"
            parts.append({"inlineData": {"mimeType": mime, "data": audio["data"]}})
        elif kind == "file":
            mime, data = _data_url(part["file"]["file_data"], where)
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        else:
            raise _refusal(f"{where}: a {kind!r} content part cannot be sent to Gemini")
    return parts


def _args_of(call_id: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = raw if isinstance(raw, str) else ""
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        raise _refusal(f"tool call {call_id!r}: its arguments are not JSON") from None
    if not isinstance(parsed, dict):
        raise _refusal(f"tool call {call_id!r}: its arguments are not a JSON object")
    return parsed


def _response_of(text: str) -> dict[str, Any]:
    """A tool result as the object `functionResponse.response` must be."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"result": text}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def contents_from(
    messages: list[Any], cache: SignatureCache, *, fold_system: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
    """`(contents, systemInstruction, replays a call without a signature)`.

    System messages are joined (Gemini takes one instruction); consecutive
    turns of one role are merged, which is also how the results of parallel
    calls end up in the single turn Google wants them in. `fold_system` is
    for models with no system instruction (Gemma): it goes in front of the
    first user turn instead.
    """
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    unsigned = False

    def add(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for index, message in enumerate(messages):
        field = f"messages[{index}].content"
        role = message.role
        if role == Role.system:
            text = "".join(p.get("text", "") for p in _parts_of(message.content, field))
            if text:
                system.append(text)
        elif role == Role.assistant:
            parts = _parts_of(message.content, field)
            for call in message.toolCalls or []:
                call_id = str(call.get("id") or "")
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                names[call_id] = name
                part: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": _args_of(call_id, function.get("arguments")),
                    }
                }
                known = cache.recall(call_id)
                if known is not None:
                    if known.signature:
                        part["thoughtSignature"] = known.signature
                    else:
                        unsigned = True
                    if known.google_id:
                        part["functionCall"]["id"] = call_id
                else:
                    unsigned = True
                parts.append(part)
            add("model", parts)
        elif role == Role.tool:
            call_id = message.toolCallId or ""
            known = cache.recall(call_id)
            name = names.get(call_id) or (known.name if known else "")
            if not name:
                raise _refusal(
                    f"messages[{index}]: the tool result for {call_id!r} matches no earlier "
                    "tool call, so Gemini cannot be told which function it answers"
                )
            text = "".join(p.get("text", "") for p in _parts_of(message.content, field))
            response: dict[str, Any] = {"name": name, "response": _response_of(text)}
            if known is not None and known.google_id:
                response["id"] = call_id
            add("user", [{"functionResponse": response}])
        else:
            add("user", _parts_of(message.content, field))
    if fold_system and system:
        lead = [{"text": "\n\n".join(system) + "\n\n"}]
        if contents and contents[0]["role"] == "user":
            contents[0]["parts"] = lead + contents[0]["parts"]
        else:
            contents.insert(0, {"role": "user", "parts": lead})
        return contents, None, unsigned
    instruction = {"parts": [{"text": "\n\n".join(system)}]} if system else None
    return contents, instruction, unsigned


def tool_config_from(choice: Any) -> dict[str, Any] | None:
    """`tool_choice` as Gemini's `toolConfig`."""
    if choice is None:
        return None
    value = getattr(choice, "value", None)
    if value is not None:
        mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}[str(value)]
        return {"functionCallingConfig": {"mode": mode}}
    return {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [choice.function.name]}
    }


def declarations_from(tools: list[Any]) -> list[dict[str, Any]]:
    """Tools as `functionDeclarations`. The JSON Schema goes as
    `parametersJsonSchema`, which takes it whole; `parameters` is an OpenAPI
    subset that refuses keywords OpenAI tools use freely."""
    declared: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.function
        item: dict[str, Any] = {"name": function.name}
        if function.description:
            item["description"] = function.description
        if function.parameters:
            item["parametersJsonSchema"] = function.parameters
        declared.append(item)
    return [{"functionDeclarations": declared}]


def thinking_config(model_id: str, effort: ReasoningEffort | None) -> dict[str, Any]:
    """`reasoning_effort` as `thinkingConfig`, by model generation.

    Gemini 3 takes a `thinkingLevel`, 2.5 a `thinkingBudget` in tokens.
    Thoughts are always asked for (`includeThoughts`) except when thinking
    is switched off, so the reasoning comes back as reasoning.
    """
    generation = generation_of(model_id)
    config: dict[str, Any] = {"includeThoughts": effort is not ReasoningEffort.none}
    if effort is None or generation is None:
        return config
    if generation >= (3, 0):
        level = {
            ReasoningEffort.none: "minimal",
            ReasoningEffort.minimal: "minimal",
            ReasoningEffort.low: "low",
            ReasoningEffort.medium: "medium",
            ReasoningEffort.high: "high",
            ReasoningEffort.xhigh: "high",
            ReasoningEffort.max: "high",
        }[effort]
        config["thinkingLevel"] = level
    else:
        pro = "pro" in model_id.lower()
        top = 32768 if pro else 24576
        budget = {
            ReasoningEffort.none: 0,
            ReasoningEffort.minimal: 128 if pro else 512,
            ReasoningEffort.low: 1024,
            ReasoningEffort.medium: 8192,
            ReasoningEffort.high: top,
            ReasoningEffort.xhigh: top,
            ReasoningEffort.max: top,
        }[effort]
        config["thinkingBudget"] = budget
    return config


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

_FINISH = {
    "STOP": FinishReason.stop,
    "MAX_TOKENS": FinishReason.length,
    "SAFETY": FinishReason.content_filter,
    "RECITATION": FinishReason.content_filter,
    "BLOCKLIST": FinishReason.content_filter,
    "PROHIBITED_CONTENT": FinishReason.content_filter,
    "SPII": FinishReason.content_filter,
    "IMAGE_SAFETY": FinishReason.content_filter,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.content_filter,
    "LANGUAGE": FinishReason.content_filter,
}
_BROKEN_CALL = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}


def finish_of(reason: Any, *, called: bool) -> FinishReason:
    """Google's `finishReason` as ours; a turn that made calls ends in them."""
    name = str(reason or "STOP")
    if name in _BROKEN_CALL:
        raise CliError(
            f"Gemini could not form the tool call it tried to make (finishReason={name}); "
            "the answer is not usable. Try again or simplify the tool's schema."
        )
    found = _FINISH.get(name, FinishReason.stop)
    return FinishReason.tool_calls if called and found is FinishReason.stop else found


def refusal_note(candidate: dict[str, Any]) -> str:
    """Google's own words for why it stopped an answer, kept with the reason."""
    reason = candidate.get("finishReason")
    message = candidate.get("finishMessage")
    return f"[Gemini stopped this answer: finishReason={reason}" + (
        f" ({message})]" if isinstance(message, str) and message else "]"
    )


def blocked_prompt(body: dict[str, Any]) -> CliError | None:
    """The refusal for a prompt Google blocked before answering, or None."""
    feedback = body.get("promptFeedback")
    if not isinstance(feedback, dict) or not feedback.get("blockReason"):
        return None
    message = feedback.get("blockReasonMessage")
    return CliError(
        f"Gemini blocked the prompt (blockReason={feedback['blockReason']}"
        + (f": {message}" if isinstance(message, str) and message else "")
        + "). Nothing was answered; rephrase the request.",
        upstream_status=400,
    )


def usage_of(raw: Any) -> Usage | None:
    """`usageMetadata` as OpenAI-style usage: completion includes the thoughts
    (Google counts them apart), and reasoning and cached are carried."""
    if not isinstance(raw, dict) or not raw:
        return None

    def whole(key: str) -> int | None:
        value = raw.get(key)
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    prompt = (whole("promptTokenCount") or 0) + (whole("toolUsePromptTokenCount") or 0)
    thoughts = whole("thoughtsTokenCount")
    completion = (whole("candidatesTokenCount") or 0) + (thoughts or 0)
    if whole("promptTokenCount") is None and whole("candidatesTokenCount") is None:
        return None
    total = whole("totalTokenCount")
    return Usage(
        promptTokens=prompt,
        completionTokens=completion,
        totalTokens=total if total is not None else prompt + completion,
        cachedPromptTokens=whole("cachedContentTokenCount"),
        reasoningTokens=thoughts,
    )


@dataclass
class Parsed:
    text: str
    reasoning: str
    calls: list[ToolCall]
    images: list[dict[str, Any]]
    audio: list[dict[str, Any]]


def read_parts(parts: Any, cache: SignatureCache, *, remember: bool = True) -> Parsed:
    """The parts of one candidate: text, thoughts, calls (remembered with
    their signatures; an id is made when Google gave none), and any media."""
    out = Parsed("", "", [], [], [])
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        call = part.get("functionCall")
        if isinstance(call, dict) and call.get("name"):
            given = call.get("id")
            call_id = str(given) if given else new_call_id()
            args = call.get("args")
            out.calls.append(
                ToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(
                        name=str(call["name"]),
                        arguments=json.dumps(args if isinstance(args, dict) else {}),
                    ),
                )
            )
            if remember:
                signature = part.get("thoughtSignature")
                cache.remember(
                    call_id,
                    RememberedCall(
                        name=str(call["name"]),
                        signature=signature if isinstance(signature, str) and signature else None,
                        google_id=bool(given),
                    ),
                )
            continue
        inline = part.get("inlineData")
        if isinstance(inline, dict) and inline.get("data"):
            mime = str(inline.get("mimeType") or "")
            (out.audio if mime.startswith("audio/") else out.images).append(inline)
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            if part.get("thought") is True:
                out.reasoning += text
            else:
                out.text += text
    return out


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

_RATE = re.compile(r"rate=(\d+)")


def pcm_rate(mime: str) -> int:
    match = _RATE.search(mime)
    return int(match.group(1)) if match else 24_000


def wav_of(pcm: bytes, rate: int = 24_000) -> bytes:
    """A complete WAV around Gemini's 16-bit mono PCM."""
    block = 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * block, block, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def decode_b64(data: str, what: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise CliError(f"Gemini's {what} was not base64") from None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

_RETRY_DELAY = re.compile(r"^(\d+(?:\.\d+)?)s$")
_REGION = re.compile(r"location is not supported|not available in your (?:country|region)", re.I)
_BAD_KEY = re.compile(r"api key (?:not valid|expired|was reported as leaked)|API_KEY_INVALID", re.I)


def _google_error(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    if isinstance(body, list) and body:
        body = body[0]
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _delay_from(error: dict[str, Any]) -> float | None:
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY.match(str(detail.get("retryDelay") or ""))
            if match:
                return float(match.group(1))
    return None


def failure_from(response: httpx.Response, what: str, *, scrub: Any = str) -> CliError:
    """Google's error JSON as the driver's failure, in the words of its cause.

    `scrub` removes the key from anything echoed. Statuses are kept so the
    gateway's taxonomy still decides: a refused key is a credential (401),
    a 429 carries its wait, a 400 stays the caller's.
    """
    error = _google_error(response)
    message = scrub(
        str(error.get("message") or response.text[:300] or f"HTTP {response.status_code}")
    )
    status_name = str(error.get("status") or "")
    status = response.status_code
    wait = retry_after(response.headers.get("Retry-After")) or _delay_from(error)
    details = json.dumps(error.get("details") or [])
    if _BAD_KEY.search(message) or _BAD_KEY.search(details):
        return CliError(
            f"Gemini refused this driver's API key for {what}: {message}",
            upstream_status=401,
        )
    if status in (401, 403) or status_name in ("PERMISSION_DENIED", "UNAUTHENTICATED"):
        return CliError(
            f"Gemini refused this driver's API key for {what} ({status_name or status}): {message}",
            upstream_status=status if status in (401, 403) else 403,
        )
    if _REGION.search(message):
        # Refused before any work, and the same for every request from here:
        # this driver's account, not the caller's request (as a 403 is).
        return CliError(
            f"Gemini is not available from where this driver runs (a region Google does not "
            f"serve) for {what}: {message}",
            upstream_status=403,
        )
    if status == 429 or status_name == "RESOURCE_EXHAUSTED":
        return CliError(
            f"Gemini's quota or rate limit was reached for {what} (RESOURCE_EXHAUSTED; the "
            f"limits are per Google project): {message}",
            upstream_status=429,
            retry_after_seconds=wait,
        )
    suffix = f" ({status_name})" if status_name else ""
    return CliError(
        f"Gemini returned {status}{suffix} for {what}: {message}",
        upstream_status=status,
        retry_after_seconds=wait,
    )


def explain_signature_refusal(error: CliError, *, replayed_unsigned: bool) -> CliError:
    """A Google refusal that names a missing thought signature, in our words."""
    if replayed_unsigned and error.upstream_status == 400 and _SIGNATURE_REFUSAL.search(str(error)):
        return CliError(
            f"{MISSING_SIGNATURE_WORDS} Google said: {str(error)[:300]}", upstream_status=400
        )
    return error

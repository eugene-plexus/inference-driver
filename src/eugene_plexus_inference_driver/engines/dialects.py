"""Provider wire behavior selected once, independently of the transport URL.

Named providers select a dialect in their registry entry. Custom endpoints may
select one explicitly; the legacy automatic mode recognizes only an exact
OpenAI hostname and otherwise retains the compatible-server protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Dialect:
    name: str
    catalogue: str = "openai"
    openai_parameters: bool = False
    max_tokens_field: str = "max_tokens"
    reasoning_key: str = "reasoning_content"
    results_in_call_order: bool = False

    @property
    def send_reasoning(self) -> bool:
        return not self.openai_parameters


DIALECTS = {
    "compatible": Dialect("compatible"),
    "openai": Dialect("openai", openai_parameters=True, max_tokens_field="max_completion_tokens"),
    "openrouter": Dialect(
        "openrouter", catalogue="openrouter", max_tokens_field="max_completion_tokens"
    ),
    "ollama": Dialect(
        "ollama", catalogue="ollama", reasoning_key="reasoning", results_in_call_order=True
    ),
    "lmstudio": Dialect("lmstudio", catalogue="lmstudio"),
}


def is_openai_endpoint(url: str) -> bool:
    try:
        return urlsplit(url).hostname in {"api.openai.com", "openai.com"}
    except ValueError:
        return False


def select(name: str | None, *, base_url: str, catalogue: str = "openai") -> Dialect:
    if name and name != "auto":
        try:
            return DIALECTS[name]
        except KeyError:
            raise ValueError(f"Unknown wire dialect {name!r}") from None
    if catalogue in {"openrouter", "ollama", "lmstudio"}:
        return DIALECTS[catalogue]
    return DIALECTS["openai" if is_openai_endpoint(base_url) else "compatible"]

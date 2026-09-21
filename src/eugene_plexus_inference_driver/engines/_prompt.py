"""Conversation -> single-prompt-string serialization for CLI adapters.

The agentic CLIs we wrap (Claude Code, Codex) take a single prompt argument
rather than a structured chat-message array, so the spec's `Message[]` has
to be collapsed into a flat tagged transcript. Approximate but adequate:
these backends are subscription passthroughs, not the local engines the
control plane is built around, and those speak OpenAI-compatible HTTP where
the message array survives intact.
"""

from __future__ import annotations

from .._generated.models import Message, Role
from ..images import content_wire
from ._subprocess import CliError


def messages_to_prompt(messages: list[Message]) -> str:
    """Render a chat-message list as a single role-prefixed transcript string."""
    lines: list[str] = []
    for msg in messages:
        label = f"[{_role_str(msg.role).upper()}]"
        lines.append(f"{label} {text_content(msg.content)}")
    return "\n\n".join(lines)


def _role_str(role: Role | str) -> str:
    return role.value if isinstance(role, Role) else str(role)


def text_content(content: object) -> str:
    value = content_wire(content)
    if isinstance(value, list):
        if any(part.get("type") != "text" for part in value):
            raise CliError("This text-only engine cannot carry image content.", upstream_status=400)
        return "".join(part["text"] for part in value)
    return value if isinstance(value, str) else ""

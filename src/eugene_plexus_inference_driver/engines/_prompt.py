"""Conversation -> single-prompt-string serialization for CLI adapters.

The bicameral CLIs we wrap (Claude Code, Codex) accept a single prompt
argument rather than a structured chat-message array. v0.1 collapses our
spec's `Message[]` into a flat tagged transcript. This is approximate but
adequate for early bicameral experiments; v0.2 may use API adapters or
session-mode CLIs to preserve true multi-turn structure.
"""

from __future__ import annotations

from .._generated.models import Message, Role


def messages_to_prompt(messages: list[Message]) -> str:
    """Render a chat-message list as a single role-prefixed transcript string."""
    lines: list[str] = []
    for msg in messages:
        role = _role_str(msg.role)
        if msg.role == Role.hemisphere and msg.driverName:
            label = f"[HEMISPHERE-{msg.driverName.upper()}]"
        elif msg.role == Role.system:
            label = "[SYSTEM]"
        elif msg.role == Role.user:
            label = "[USER]"
        elif msg.role == Role.assistant:
            label = "[ASSISTANT]"
        else:
            label = f"[{role.upper()}]"
        lines.append(f"{label} {msg.content}")
    return "\n\n".join(lines)


def _role_str(role: Role | str) -> str:
    return role.value if isinstance(role, Role) else str(role)

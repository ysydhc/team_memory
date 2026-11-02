"""Platform-specific stdout output helpers. 平台特定的 stdout 输出适配器。"""

import json
import sys


def print_json(data: dict) -> None:
    """Print JSON to stdout and flush. 输出 JSON 到 stdout 并 flush。"""
    print(json.dumps(data))
    sys.stdout.flush()


def print_empty() -> None:
    """Print empty JSON response. 输出空 JSON 响应。"""
    print("{}")
    sys.stdout.flush()


def print_claude_context(text: str) -> None:
    """Print Claude Code additionalContext response. 输出 Claude Code 上下文响应。"""
    if text:
        print_json({"additionalContext": text})
    else:
        print_empty()


def print_cursor_action(data: dict) -> None:
    """Print Cursor action response. 输出 Cursor action 响应。"""
    print_json(data)


def print_hermes_context(text: str) -> None:
    """Print Hermes context response. 输出 Hermes 上下文响应。"""
    if text:
        print_json({"context": text})
    else:
        print_empty()


def print_antigravity_system_message(text: str) -> None:
    """Print Antigravity SessionStart/PreInvocation systemMessage injection."""
    if text:
        print_json({"injectSteps": [{"systemMessage": {"systemMessage": text}}]})
    else:
        print_empty()


def print_antigravity_ephemeral_message(text: str) -> None:
    """Print Antigravity SessionStart/PreInvocation ephemeralMessage injection."""
    if text:
        print_json({"injectSteps": [{"ephemeralMessage": text}]})
    else:
        print_empty()


def print_antigravity_stop(decision: str = "allow") -> None:
    """Print Antigravity Stop hook decision."""
    print_json({"decision": decision})

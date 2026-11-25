"""CLI argument parser builder and main() entrypoint."""

from __future__ import annotations

import argparse

from team_memory.cli.parser_core import register_core_subparsers
from team_memory.cli.parser_ops import register_ops_subparsers


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level tm-cli ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="tm-cli",
        description="Team Memory CLI — archives, uploads, and MCP HTTP helpers",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_core_subparsers(sub)
    register_ops_subparsers(sub)
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

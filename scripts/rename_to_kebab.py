#!/usr/bin/env python3
"""Convert camelCase/snake_case filenames to kebab-case. Run from repo root."""

import re
from pathlib import Path


def to_kebab(s: str) -> str:
    """Convert to kebab-case."""
    if not s:
        return s
    if s.upper() in ("README", "README.MD"):
        return "README.md" if s.upper() == "README.MD" else "README"
    if "." in s and s.rsplit(".", 1)[-1].lower() in ("md", "yaml", "yml"):
        name, ext = s.rsplit(".", 1)
        return to_kebab(name) + "." + ext
    # Insert hyphen before uppercase, then lowercase
    s = re.sub(r"([a-z])([A-Z])", r"\1-\2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    s = re.sub(r"_+", "-", s)
    return s.lower()


def main():
    root = Path(__file__).resolve().parent.parent / "docs"
    renames = []
    for p in sorted(root.rglob("*"), key=lambda x: (-len(x.parts), str(x))):
        if not p.is_file():
            continue
        name = p.name
        new_name = to_kebab(name)
        if new_name != name:
            new_path = p.parent / new_name
            renames.append((p, new_path))
    for old, new in renames:
        print(f"mv|{old}|{new}")


if __name__ == "__main__":
    main()

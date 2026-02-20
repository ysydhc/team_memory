#!/usr/bin/env sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PACK_DIR="$ROOT_DIR/.debug/knowledge-pack"
RULE_SRC="$PACK_DIR/rules/team_memory-codified-shortcuts.mdc"
SKILL_SRC="$PACK_DIR/skills/team-memory-codified-workflow/SKILL.md"

RULE_DST_DIR="$ROOT_DIR/.cursor/rules"
RULE_DST="$RULE_DST_DIR/team_memory-codified-shortcuts.mdc"

SKILL_DST_DIR="$ROOT_DIR/.cursor/skills/team-memory-codified-workflow"
SKILL_DST="$SKILL_DST_DIR/SKILL.md"

if [ ! -f "$RULE_SRC" ]; then
  echo "ERROR: rule source not found: $RULE_SRC"
  exit 1
fi

if [ ! -f "$SKILL_SRC" ]; then
  echo "ERROR: skill source not found: $SKILL_SRC"
  exit 1
fi

mkdir -p "$RULE_DST_DIR"
mkdir -p "$SKILL_DST_DIR"

cp "$RULE_SRC" "$RULE_DST"
cp "$SKILL_SRC" "$SKILL_DST"

echo "Installed codified knowledge pack:"
echo "  - $RULE_DST"
echo "  - $SKILL_DST"
echo ""
echo "Done. Reopen Cursor session (or reload rules/skills) to ensure immediate effect."

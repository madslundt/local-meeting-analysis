#!/usr/bin/env bash
# Explore generated meeting documents in a grounded, multi-turn local chat.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

require_python
source "$ROOT/bin/lib/llm_serve.sh"

case "${1:-}" in
  -h|--help)
    "$PY" "$ROOT/bin/lib/chat_docs.py" --help
    exit 0
    ;;
esac

llm_ensure
NUM_CTX="$NUM_CTX" CHAT_RESERVE_TOKENS="$CHAT_RESERVE_TOKENS" \
CHAT_TEMPERATURE="$CHAT_TEMPERATURE" LLM_HOST_URL="$LLM_HOST_URL" \
"$PY" "$ROOT/bin/lib/chat_docs.py" "$@"

llm_note_resident

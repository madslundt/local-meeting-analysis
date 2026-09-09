#!/usr/bin/env bash
# Stage 2: transcripts/<name>_transcript.txt  ->  summaries/<name>_summary.md
#
#   bin/summarize.sh [--fresh]
#
# A context-free record of what happened: who was there, what was covered, what
# was actually said. No judgement and no reference to your criteria -- that is
# stage 3's job. Keeping them apart means the summary stays useful even if you
# later change your mind about what you want.
#
# Prefers the labelled transcript and falls back to the raw SPEAKER_n transcript.
# Current summaries are skipped; summaries older than their selected transcript
# or prompts are rebuilt automatically. --fresh redoes all.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

usage() {
  cat <<'EOF'
Usage: bin/summarize.sh [--fresh]

  --fresh  Regenerate every summary from the current transcripts.
  -h, --help
           Show this help.

FORCE=1 remains supported as an environment-variable equivalent to --fresh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FORCE=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

require_python
source "$ROOT/bin/lib/llm_serve.sh"

shopt -s nullglob
FILES=("$TRANSCRIPTS"/*_transcript.txt)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No transcripts in $TRANSCRIPTS -- run bin/transcribe.sh first." >&2
  exit 1
fi

done_count=0 skip_count=0
llm_started=0
for RAW_T in "${FILES[@]}"; do
  [[ "$RAW_T" == *_labelled_transcript.txt ]] && continue
  BASE="$(basename "$RAW_T" .txt)"          # <name>_transcript
  NAME="${BASE%_transcript}"
  LABELLED_T="$TRANSCRIPTS/${NAME}_labelled_transcript.txt"
  if [[ -f "$LABELLED_T" ]]; then
    T="$LABELLED_T"
  else
    T="$RAW_T"
    echo "-- $NAME: no labelled transcript; using SPEAKER_n labels"
  fi
  OUT="$SUMMARIES/${NAME}_summary.md"

  if [[ -f "$OUT" && $FORCE -eq 0 \
        && "$OUT" -nt "$T" \
        && "$OUT" -nt "$PROMPTS/summary.system.md" \
        && "$OUT" -nt "$PROMPTS/summary.prompt.md" ]]; then
    echo "-- $NAME: summary is current (--fresh to redo)"
    skip_count=$((skip_count + 1))
    continue
  fi

  if [[ -f "$OUT" && $FORCE -eq 0 ]]; then
    echo "-- $NAME: inputs changed; rebuilding summary"
  fi

  echo
  echo "== $NAME"
  if [[ $llm_started -eq 0 ]]; then
    llm_ensure
    llm_started=1
  fi
  NUM_CTX="$NUM_CTX" RESERVE_TOKENS="$RESERVE_TOKENS" LLM_HOST_URL="$LLM_HOST_URL" \
  "$PY" "$ROOT/bin/lib/llm_run.py" \
    --transcript "$T" \
    --system   "$PROMPTS/summary.system.md" \
    --template "$PROMPTS/summary.prompt.md" \
    --temperature "$SUMMARY_TEMPERATURE" \
    --out "$OUT"
  done_count=$((done_count + 1))
done

echo
echo "summarised $done_count, skipped $skip_count -> $(basename "$SUMMARIES")/"
[[ $llm_started -eq 0 ]] || llm_note_resident

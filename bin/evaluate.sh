#!/usr/bin/env bash
# Stage 3: transcript + your context  ->  evaluations/<name>_evaluation.md
#
#   bin/evaluate.sh [--fresh]
#
# Context resolution, in order:
#   1. CONTEXT_FILE from config.sh, if set -- one complete context for every recording
#   2. Otherwise contexts/default.md is the shared criteria, optionally followed
#      by contexts/<recording name>.md with facts specific to that meeting.
#
# Reads the labelled transcript when available, otherwise the raw transcript.
# It does not read the summary: a summary of a summary loses the detail and the
# timestamps that make a citation checkable.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

usage() {
  cat <<'EOF'
Usage: bin/evaluate.sh [--fresh]

  --fresh  Regenerate every evaluation from the current transcripts and context.
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
input_count=0
llm_started=0
for RAW_T in "${FILES[@]}"; do
  [[ "$RAW_T" == *_labelled_transcript.txt ]] && continue
  input_count=$((input_count + 1))
  BASE="$(basename "$RAW_T" .txt)"
  NAME="${BASE%_transcript}"
  LABELLED_T="$TRANSCRIPTS/${NAME}_labelled_transcript.txt"
  if [[ -f "$LABELLED_T" ]]; then
    T="$LABELLED_T"
  else
    T="$RAW_T"
    echo "-- $NAME: no labelled transcript; using SPEAKER_n labels"
  fi
  OUT="$EVALUATIONS/${NAME}_evaluation.md"

  BUILD_COMBINED_CONTEXT=0
  if [[ -n "$CONTEXT_FILE" ]]; then
    CTX="$CONTEXT_FILE"
    CTX_INPUTS=("$CONTEXT_FILE")
  else
    SHARED_CTX="$CONTEXT_DEFAULT"
    SPECIFIC_CTX="$CONTEXTS/$NAME.md"
    if [[ -f "$SHARED_CTX" && -f "$SPECIFIC_CTX" && "$SHARED_CTX" != "$SPECIFIC_CTX" ]]; then
      CTX="$WORK/$NAME.context.md"
      CTX_INPUTS=("$SHARED_CTX" "$SPECIFIC_CTX")
      BUILD_COMBINED_CONTEXT=1
    elif [[ -f "$SPECIFIC_CTX" ]]; then
      CTX="$SPECIFIC_CTX"
      CTX_INPUTS=("$SPECIFIC_CTX")
    else
      CTX="$SHARED_CTX"
      CTX_INPUTS=("$SHARED_CTX")
    fi
  fi

  MISSING_CONTEXT=0
  for CTX_INPUT in "${CTX_INPUTS[@]}"; do
    [[ -f "$CTX_INPUT" ]] || MISSING_CONTEXT=1
  done
  if [[ $MISSING_CONTEXT -eq 1 ]]; then
    echo "-- $NAME: no context file found." >&2
    echo "   Looked for shared criteria at $CONTEXT_DEFAULT and meeting context at $CONTEXTS/$NAME.md" >&2
    echo "   Copy contexts/example.md and fill it in." >&2
    exit 1
  fi

  OUTPUT_CURRENT=0
  if [[ -f "$OUT" && $FORCE -eq 0 \
        && "$OUT" -nt "$T" \
        && "$OUT" -nt "$PROMPTS/evaluation.system.md" \
        && "$OUT" -nt "$PROMPTS/evaluation.prompt.md" ]]; then
    OUTPUT_CURRENT=1
    for CTX_INPUT in "${CTX_INPUTS[@]}"; do
      [[ "$OUT" -nt "$CTX_INPUT" ]] || OUTPUT_CURRENT=0
    done
  fi
  if [[ $OUTPUT_CURRENT -eq 1 ]]; then
    echo "-- $NAME: evaluation is current (--fresh to redo)"
    skip_count=$((skip_count + 1))
    continue
  fi

  if [[ -f "$OUT" && $FORCE -eq 0 ]]; then
    echo "-- $NAME: inputs changed; rebuilding evaluation"
  fi

  if [[ $BUILD_COMBINED_CONTEXT -eq 1 ]]; then
    {
      printf '# Shared decision criteria\n\n'
      cat "${CTX_INPUTS[0]}"
      printf '\n\n# Context specific to this meeting\n\n'
      cat "${CTX_INPUTS[1]}"
    } > "$CTX"
  fi

  echo
  echo "== $NAME  (context: $(basename "$CTX"))"
  if [[ $llm_started -eq 0 ]]; then
    llm_ensure
    llm_started=1
  fi
  NUM_CTX="$NUM_CTX" RESERVE_TOKENS="$RESERVE_TOKENS" LLM_HOST_URL="$LLM_HOST_URL" \
  "$PY" "$ROOT/bin/lib/llm_run.py" \
    --transcript "$T" \
    --context "$CTX" \
    --system   "$PROMPTS/evaluation.system.md" \
    --template "$PROMPTS/evaluation.prompt.md" \
    --temperature "$EVALUATION_TEMPERATURE" \
    --out "$OUT"
  done_count=$((done_count + 1))
done

echo
echo "evaluated $done_count, skipped $skip_count -> $(basename "$EVALUATIONS")/"
if [[ $input_count -gt 1 ]]; then
  echo "Next: bin/compare.sh, to compare them against each other."
fi
[[ $llm_started -eq 0 ]] || llm_note_resident

#!/usr/bin/env bash
# Stage 4: all evaluations + your criteria  ->  comparisons/comparison.md
#
#   bin/compare.sh [--fresh]
#
# The other stages judge one meeting at a time. This compares the
# evaluated subjects against one criteria set and isolates meaningful differences.
#
# Reads the evaluations rather than the transcripts: three 50-minute transcripts
# is roughly 39k tokens, past the window, and weighing conclusions is the job
# here. Where it needs a quote it can take the one the evaluation already cited.
#
# This needs ONE shared criteria set -- ranking against three different
# yardsticks is meaningless -- so it uses CONTEXT_FILE or contexts/default.md
# and ignores the per-recording contexts that bin/evaluate.sh honours.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

usage() {
  cat <<'EOF'
Usage: bin/compare.sh [--fresh]

  --fresh  Regenerate the comparison from the current evaluations and context.
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
FILES=("$EVALUATIONS"/*_evaluation.md)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No evaluations in $EVALUATIONS -- run bin/evaluate.sh first." >&2
  exit 1
fi
if [[ ${#FILES[@]} -eq 1 ]]; then
  echo "Only one evaluation ($(basename "${FILES[0]}")). There is nothing to" >&2
  echo "compare it against -- read it directly instead." >&2
  exit 1
fi

OUT="$COMPARISONS/comparison.md"
if [[ -n "$CONTEXT_FILE" ]]; then
  CTX="$CONTEXT_FILE"
else
  CTX="$CONTEXT_DEFAULT"
fi
if [[ ! -f "$CTX" ]]; then
  echo "no shared context file at $CTX" >&2
  echo "Comparison needs one criteria set for all evaluated subjects. Copy" >&2
  echo "contexts/example.md to contexts/default.md and fill it in," >&2
  echo "or point CONTEXT_FILE at the one you want to rank against." >&2
  exit 1
fi

# Comparison consumes evaluations rather than several full transcripts. Verify
# that each evaluation used the currently preferred transcript (labelled when
# available, raw otherwise). This later stage never rebuilds an earlier one.
STALE_EVALUATIONS=0
for E in "${FILES[@]}"; do
  NAME="$(basename "$E" .md)"; NAME="${NAME%_evaluation}"
  LABELLED_T="$TRANSCRIPTS/${NAME}_labelled_transcript.txt"
  RAW_T="$TRANSCRIPTS/${NAME}_transcript.txt"
  if [[ -f "$LABELLED_T" ]]; then
    SOURCE_T="$LABELLED_T"
    SOURCE_KIND="labelled transcript"
  elif [[ -f "$RAW_T" ]]; then
    SOURCE_T="$RAW_T"
    SOURCE_KIND="raw transcript"
  else
    echo "-- $NAME: evaluation has no matching transcript" >&2
    STALE_EVALUATIONS=1
    continue
  fi

  EVALUATION_STALE=0
  [[ "$E" -nt "$SOURCE_T" ]] || EVALUATION_STALE=1
  [[ "$E" -nt "$PROMPTS/evaluation.system.md" ]] || EVALUATION_STALE=1
  [[ "$E" -nt "$PROMPTS/evaluation.prompt.md" ]] || EVALUATION_STALE=1
  [[ "$E" -nt "$CTX" ]] || EVALUATION_STALE=1
  SPECIFIC_CTX="$CONTEXTS/$NAME.md"
  if [[ -z "$CONTEXT_FILE" && -f "$SPECIFIC_CTX" ]]; then
    [[ "$E" -nt "$SPECIFIC_CTX" ]] || EVALUATION_STALE=1
  fi
  if [[ $EVALUATION_STALE -eq 1 ]]; then
    echo "-- $NAME: evaluation is older than its $SOURCE_KIND or evaluation inputs" >&2
    STALE_EVALUATIONS=1
  fi
done
if [[ $STALE_EVALUATIONS -eq 1 ]]; then
  echo "Run bin/evaluate.sh first; comparison will not modify an earlier stage." >&2
  exit 1
fi

OUTPUT_CURRENT=0
if [[ -f "$OUT" && $FORCE -eq 0 \
      && "$OUT" -nt "$CTX" \
      && "$OUT" -nt "$PROMPTS/comparison.system.md" \
      && "$OUT" -nt "$PROMPTS/comparison.prompt.md" \
      && "$OUT" -nt "$EVALUATIONS" ]]; then
  OUTPUT_CURRENT=1
  for E in "${FILES[@]}"; do
    [[ "$OUT" -nt "$E" ]] || OUTPUT_CURRENT=0
  done
fi
if [[ $OUTPUT_CURRENT -eq 1 ]]; then
  echo "$(basename "$OUT") is current (--fresh to redo)"
  exit 0
fi
if [[ -f "$OUT" && $FORCE -eq 0 ]]; then
  echo "comparison inputs changed; rebuilding"
fi

# One file, with each evaluation under a heading so the model can name them.
BUNDLE="$WORK/all-evaluations.md"
: > "$BUNDLE"
echo "comparing ${#FILES[@]} evaluations:"
for E in "${FILES[@]}"; do
  NAME="$(basename "$E" .md)"; NAME="${NAME%_evaluation}"
  echo "  - $NAME"
  { printf '\n\n# Evaluation: %s\n\n' "$NAME"; cat "$E"; } >> "$BUNDLE"
done

llm_ensure

echo
echo "== comparison  (criteria: $(basename "$CTX"))"
NUM_CTX="$NUM_CTX" RESERVE_TOKENS="$RESERVE_TOKENS" LLM_HOST_URL="$LLM_HOST_URL" \
"$PY" "$ROOT/bin/lib/llm_run.py" \
  --evaluations "$BUNDLE" \
  --context "$CTX" \
  --system   "$PROMPTS/comparison.system.md" \
  --template "$PROMPTS/comparison.prompt.md" \
  --temperature "$COMPARISON_TEMPERATURE" \
  --out "$OUT"

echo
echo "-> $OUT"
llm_note_resident

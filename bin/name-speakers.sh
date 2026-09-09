#!/usr/bin/env bash
# Stage 1b: put names to SPEAKER_0/1/2, then rebuild the transcripts.
#
#   bin/name-speakers.sh [--fresh]
#
# Interactive: it shows you each voice and asks who it is. Recordings that
# already have a speakers/<name>.json are skipped; --fresh revisits them (your
# existing answers are offered as defaults).
#
# Only the merge step runs afterwards, creating a separate labelled transcript
# beside the raw transcript. The raw transcript is never modified.
#
# Do this BEFORE summarising. Correct names and roles keep each participant's
# statements attached to the right person.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

usage() {
  cat <<'EOF'
Usage: bin/name-speakers.sh [--fresh]

  --fresh  Revisit every speaker map and rebuild every transcript.
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

shopt -s nullglob
DIARS=("$WORK"/*.diarization.json)

if [[ ${#DIARS[@]} -eq 0 ]]; then
  echo "Nothing to name -- run bin/transcribe.sh first." >&2
  exit 1
fi

named=0 skipped=0
for DIAR in "${DIARS[@]}"; do
  BASE="$(basename "$DIAR" .diarization.json)"
  ASR_JSON="$WORK/$BASE.asr.json"
  LEGACY_WHISPER_JSON="$WORK/$BASE.whisper.json"
  AUDIO="$WORK/$BASE.16k.wav"
  MAP="$SPEAKERMAPS/$BASE.json"
  OUT="$TRANSCRIPTS/${BASE}_labelled_transcript.txt"

  if [[ ! -f "$ASR_JSON" ]]; then
    if [[ -f "$LEGACY_WHISPER_JSON" ]]; then
      # Before multiple ASR backends were supported, the same JSON format used
      # a Whisper-specific filename. Keep those expensive caches reusable.
      ASR_JSON="$LEGACY_WHISPER_JSON"
    else
      echo "-- $BASE: no cached ASR output; re-run bin/transcribe.sh" >&2
      continue
    fi
  fi
  if [[ -f "$MAP" && $FORCE -eq 0 ]]; then
    if [[ -f "$OUT" && "$OUT" -nt "$MAP" && "$OUT" -nt "$ASR_JSON" && "$OUT" -nt "$DIAR" ]]; then
      echo "-- $BASE: labels already applied (--fresh to revisit)"
      skipped=$((skipped + 1))
      continue
    fi
    echo "-- $BASE: rebuilding transcript from existing labels"
  else
    if [[ ! -f "$AUDIO" ]]; then
      echo "-- $BASE: no cached audio; re-run bin/transcribe.sh" >&2
      continue
    fi

    "$PY" "$ROOT/bin/lib/name_speakers.py" \
      --asr-json "$ASR_JSON" --diarization-json "$DIAR" \
      --audio "$AUDIO" --out "$MAP" --label "$BASE" \
      --window "$NAME_WINDOW_SECONDS" --samples "$NAME_SAMPLES" \
      --max-audio-seconds "$NAME_AUDIO_SECONDS"
  fi

  OUT_PART="$OUT.part"
  "$PY" "$ROOT/bin/lib/merge.py" \
    --asr-json "$ASR_JSON" --diarization-json "$DIAR" \
    --speakers-map "$MAP" > "$OUT_PART"
  mv "$OUT_PART" "$OUT"

  echo
  echo "rebuilt $(basename "$OUT"). It now opens:"
  head -n 5 "$OUT" | sed 's/^/    /'
  named=$((named + 1))
done

echo
echo "named $named, skipped $skipped"
if [[ $named -gt 0 ]]; then
  cat <<'EOF'

Skim a transcript before going on. If a name is attached to the wrong voice,
re-run with --fresh -- your previous answers come back as defaults.

Next: bin/summarize.sh
EOF
fi

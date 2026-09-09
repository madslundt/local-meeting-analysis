#!/usr/bin/env bash
# Stage 1: recordings/*.{mp3,wav,m4a}  ->  transcripts/<name>_transcript.txt
#
#   bin/transcribe.sh [--fresh]
#
# All settings come from config.sh. Already-transcribed recordings are skipped,
# so re-running only picks up new files; pass --fresh to redo everything.
#
# Intermediates land in work/ and are reused. This stage always writes an
# immutable raw transcript with SPEAKER_n labels; bin/name-speakers.sh creates
# a separate labelled transcript rather than replacing it.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

usage() {
  cat <<'EOF'
Usage: bin/transcribe.sh [--fresh]

  --fresh  Regenerate every transcript and all cached transcription stages.
  -h, --help
           Show this help.

FORCE=1 remains supported as an environment-variable equivalent to --fresh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FORCE=1
      ;;
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

for f in "$SEG_MODEL" "$EMB_MODEL"; do
  [[ -f "$f" ]] || { echo "missing model: $f" >&2
                     echo "run bin/fetch-models.sh first" >&2; exit 1; }
done
command -v ffmpeg      >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }
case "$ASR_BACKEND" in
  whisper-cpp)
    [[ -f "$GGML" ]] || { echo "missing model: $GGML" >&2; echo "run bin/fetch-models.sh first" >&2; exit 1; }
    command -v whisper-cli >/dev/null || { echo "whisper-cli not found (brew install whisper-cpp)" >&2; exit 1; }
    ;;
  huggingface)
    if [[ -d "$ASR_MODEL" ]]; then
      HF_MODEL_REF="$ASR_MODEL"
    else
      HF_MODEL_REF="$HF_MODEL_DIR"
      [[ -f "$HF_MODEL_MARKER" ]] || { echo "missing or incomplete Hugging Face model: $ASR_MODEL@$HF_REVISION" >&2; echo "run bin/fetch-models.sh first" >&2; exit 1; }
    fi
    ;;
  *) echo "invalid ASR_BACKEND '$ASR_BACKEND' (expected whisper-cpp or huggingface)" >&2; exit 2 ;;
esac

shopt -s nullglob nocaseglob
FILES=("$RECORDINGS"/*.mp3 "$RECORDINGS"/*.wav "$RECORDINGS"/*.m4a "$RECORDINGS"/*.mp4 "$RECORDINGS"/*.flac)
shopt -u nocaseglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No audio in $RECORDINGS -- put your .mp3/.wav files there." >&2
  exit 1
fi

echo "${#FILES[@]} recording(s) in $(basename "$RECORDINGS")/"
done_count=0 skip_count=0

for AUDIO in "${FILES[@]}"; do
  BASE="$(basename "$AUDIO")"; BASE="${BASE%.*}"
  OUT="$TRANSCRIPTS/${BASE}_transcript.txt"

  echo
  echo "== $BASE"
  WAV="$WORK/$BASE.16k.wav"
  DIAR="$WORK/$BASE.diarization.json"
  ASR_JSON="$WORK/$BASE.asr.json"
  ASR_META="$WORK/$BASE.asr.meta"

  SPEAKER_COUNT="$SPEAKERS"
  COUNT_FILE="$SPEAKERMAPS/$BASE.count"
  if [[ -f "$COUNT_FILE" ]]; then
    read -r SPEAKER_COUNT < "$COUNT_FILE"
  fi
  [[ "$SPEAKER_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "   invalid speaker count '$SPEAKER_COUNT' (expected a positive integer)" >&2
    echo "   fix $COUNT_FILE or SPEAKERS in config.sh" >&2
    exit 1
  }
  ASR_CACHE_KEY="$ASR_BACKEND|$ASR_MODEL|$AUDIO_LANG|$HF_REVISION|$WHISPER_QUANT|speakers=$SPEAKER_COUNT"

  # A cached result made with a different cluster count is not reusable.
  DIAR_SPEAKERS=""
  if [[ -f "$DIAR" ]]; then
    DIAR_SPEAKERS="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("speakers", ""))' "$DIAR" 2>/dev/null || true)"
  fi
  CACHED_ASR_KEY=""
  [[ -f "$ASR_META" ]] && read -r CACHED_ASR_KEY < "$ASR_META"
  if [[ -f "$OUT" && -f "$ASR_JSON" && $FORCE -eq 0 \
        && "$DIAR_SPEAKERS" == "$SPEAKER_COUNT" \
        && "$CACHED_ASR_KEY" == "$ASR_CACHE_KEY" ]]; then
    echo "-- $BASE: already transcribed with $ASR_MODEL and $SPEAKER_COUNT speakers (--fresh to redo)"
    skip_count=$((skip_count + 1))
    continue
  fi

  # 1. Both whisper.cpp and the diarizer require 16 kHz mono 16-bit PCM.
  if [[ ! -f "$WAV" || $FORCE -eq 1 ]]; then
    echo "   converting to 16 kHz mono"
    ffmpeg -nostdin -loglevel error -y -i "$AUDIO" \
      -ac 1 -ar 16000 -c:a pcm_s16le "$WAV"
  fi
  MINS="$("$PY" -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate()/60:.1f}')" "$WAV")"
  echo "   ${MINS} min"

  # 2. Who spoke when -- independent of what was said.
  if [[ ! -f "$DIAR" || $FORCE -eq 1 || "$DIAR_SPEAKERS" != "$SPEAKER_COUNT" ]]; then
    if [[ -f "$DIAR" && $FORCE -eq 0 ]]; then
      echo "   cached diarization used ${DIAR_SPEAKERS:-an unknown number of} speakers; rebuilding"
    fi
    echo "   diarizing ($SPEAKER_COUNT speakers)"
    DIAR_PART="$DIAR.part"
    "$PY" "$ROOT/bin/lib/diarize.py" "$WAV" \
      --segmentation "$SEG_MODEL" --embedding "$EMB_MODEL" \
      --speakers "$SPEAKER_COUNT" > "$DIAR_PART"
    mv "$DIAR_PART" "$DIAR"
  fi

  # 3. What was said. Both backends produce the same timestamped JSON shape.
  if [[ ! -f "$ASR_JSON" || $FORCE -eq 1 || "$CACHED_ASR_KEY" != "$ASR_CACHE_KEY" ]]; then
    echo "   transcribing with $ASR_MODEL (backend=$ASR_BACKEND, lang=$AUDIO_LANG)"
    if [[ "$ASR_BACKEND" == "whisper-cpp" ]]; then
      TMP_PREFIX="$WORK/$BASE.asr-tmp"
      whisper-cli -m "$GGML" -f "$WAV" -l "$AUDIO_LANG" -bs 5 -mc 0 -pp \
        --output-json --output-file "$TMP_PREFIX" >/dev/null
      [[ -f "$TMP_PREFIX.json" ]] || { echo "   whisper wrote no JSON" >&2; exit 1; }
      mv "$TMP_PREFIX.json" "$ASR_JSON"
    else
      HF_ARGS=(
        --audio "$WAV" --diarization "$DIAR" --model "$HF_MODEL_REF"
        --model-id "$ASR_MODEL" --language "$AUDIO_LANG" --device "$HF_DEVICE"
        --batch-size "$HF_BATCH_SIZE" --revision "$HF_REVISION" --output "$ASR_JSON"
      )
      [[ "$HF_TRUST_REMOTE_CODE" == "1" ]] && HF_ARGS+=(--trust-remote-code)
      "$PY" "$ROOT/bin/lib/transcribe_hf.py" "${HF_ARGS[@]}"
    fi
    printf '%s\n' "$ASR_CACHE_KEY" > "$ASR_META.part"
    mv "$ASR_META.part" "$ASR_META"
  fi

  # 4. Stitch together the raw transcript. Speaker naming is a separate stage
  # and must never overwrite this evidence-preserving output.
  OUT_PART="$OUT.part"
  "$PY" "$ROOT/bin/lib/merge.py" \
    --asr-json "$ASR_JSON" --diarization-json "$DIAR" > "$OUT_PART"
  mv "$OUT_PART" "$OUT"

  echo "   -> $(basename "$OUT")"
  done_count=$((done_count + 1))
done

echo
echo "transcribed $done_count, skipped $skip_count -> $(basename "$TRANSCRIPTS")/"
cat <<'EOF'

Next: bin/name-speakers.sh

It shows you each voice and asks who it is, then rebuilds the transcripts with
real names. Do it before summarising so each statement remains attached to the
right participant.
EOF

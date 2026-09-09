#!/usr/bin/env bash
# Acquire every model the pipeline needs, skipping whatever is already present.
# Safe to re-run; each step is a no-op once satisfied.
#
#   bin/fetch-models.sh                # download whatever is missing
#   CHECK_ONLY=1 bin/fetch-models.sh   # report only, download nothing
#
# Speech models can come from OpenAI's checkpoint store or Hugging Face. The
# evaluator GGUF comes from ModelScope. Downloads are kept in this project:
#   openaipublic.azureedge.net  - OpenAI's own Whisper checkpoints
#   huggingface.co              - optional Transformers ASR models
#   www.modelscope.ai           - Alibaba's international model hub endpoint
#   github.com releases         - sherpa-onnx diarization models

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

: "${CHECK_ONLY:=0}"
if [[ $CHECK_ONLY -eq 0 ]]; then
  require_python
fi

ok()   { printf '  \033[32mok\033[0m      %s\n' "$1"; }
todo() { printf '  \033[33mmissing\033[0m %s\n' "$1"; }
die()  { printf '\033[31merror\033[0m %s\n' "$1" >&2; exit 1; }

# A proxy block page arrives as HTTP 200 with text/html. Treating that as a
# successful download is exactly how a 20 GB pull once produced garbage here,
# so nothing is trusted until it has been inspected.
verify_binary() {
  local path="$1" want_magic="${2:-}"
  [[ -s "$path" ]] || die "$path is empty"
  if head -c 512 "$path" | LC_ALL=C grep -qi '<!doctype html\|<html'; then
    die "$path is an HTML block page, not a model -- the host is intercepted."
  fi
  if [[ -n "$want_magic" ]]; then
    local got
    got="$(head -c 4 "$path" | LC_ALL=C tr -d '\0')"
    [[ "$got" == "$want_magic" ]] || die "$path: expected magic '$want_magic', got '$got'"
  fi
}

# fetch <url> <dest> [expected-magic]
#
# Downloads to <dest>.part and only moves it into place once curl reports
# success, so a file at its final name is always a complete file. Without this
# an interrupted 20 GB download leaves a fragment that every later "is it
# present?" check happily accepts -- which is a silent failure of precisely the
# kind this pipeline is trying to avoid. Resumable: re-running continues the
# existing .part rather than starting over.
fetch() {
  local url="$1" dest="$2" magic="${3:-}" part="$2.part"
  if [[ -f "$part" ]]; then
    echo "    resuming $(basename "$dest") from $(du -h "$part" | cut -f1)"
  else
    echo "    downloading $(basename "$dest")"
  fi
  curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$part" "$url" \
    || die "download failed (re-run to resume): $url"
  verify_binary "$part" "$magic"
  mv "$part" "$dest"
}

# ---------------------------------------------------------------------------
# 1. Speech recognition model
# ---------------------------------------------------------------------------
# The checkpoint URLs and their content hashes live in openai/whisper's
# __init__.py. Read them from that source of truth rather than hardcoding a
# hash here: any variant then works, and no URL is guessed.
resolve_whisper_url() {
  local variant="$1"
  curl -fsSL "https://raw.githubusercontent.com/openai/whisper/main/whisper/__init__.py" \
    | "$PY" -c "
import re,sys
src=sys.stdin.read()
m=re.search(r'_MODELS\s*=\s*\{(.*?)\n\}', src, re.S)
if not m: sys.exit('could not parse _MODELS from whisper/__init__.py')
urls=dict(re.findall(r'\"([\w.\-]+)\"\s*:\s*\"(https://[^\"]+)\"', m.group(1)))
u=urls.get('$variant')
if not u: sys.exit('unknown whisper.cpp ASR_MODEL \'$variant\'; available: '+', '.join(sorted(urls)))
print(u)
"
}

case "$ASR_BACKEND" in
whisper-cpp)
  echo "Speech recognition: whisper.cpp ($ASR_MODEL)"
  PT="$MODELS/$ASR_MODEL.pt"
  if [[ -f "$GGML" ]]; then
    ok "$(basename "$GGML")"
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    todo "$(basename "$GGML")"
  else
    todo "$(basename "$GGML")"
    if [[ ! -f "$PT" ]]; then
      URL="$(resolve_whisper_url "$ASR_MODEL")" || die "cannot resolve checkpoint URL"
      fetch "$URL" "$PT"
    fi

    # Convert the PyTorch checkpoint to whisper.cpp's GGML format.
    CONV="$WORK/convert"
    mkdir -p "$CONV/whisper-repo/whisper/assets"
    [[ -f "$CONV/convert-pt-to-ggml.py" ]] || curl -fsSL \
      -o "$CONV/convert-pt-to-ggml.py" \
      "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/models/convert-pt-to-ggml.py"
    for a in mel_filters.npz multilingual.tiktoken; do
      [[ -f "$CONV/whisper-repo/whisper/assets/$a" ]] || curl -fsSL \
        -o "$CONV/whisper-repo/whisper/assets/$a" \
        "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/$a"
    done

    echo "    converting to ggml (keep enough free RAM for the selected checkpoint)"
    rm -rf "$CONV/out" && mkdir -p "$CONV/out"
    "$PY" "$CONV/convert-pt-to-ggml.py" "$PT" "$CONV/whisper-repo" "$CONV/out" \
      || die "ggml conversion failed"
    found="$(find "$CONV/out" -name '*.bin' -print -quit)"
    [[ -n "$found" ]] || die "conversion produced no .bin file"
    mv "$found" "$GGML_BASE"

    if [[ -n "$WHISPER_QUANT" ]]; then
      echo "    quantising to $WHISPER_QUANT"
      whisper-quantize "$GGML_BASE" "$GGML" "$WHISPER_QUANT" || die "quantise failed"
    fi
    ok "$(basename "$GGML")"
  fi
  ;;
huggingface)
  echo "Speech recognition: Hugging Face ($ASR_MODEL)"
  if [[ -d "$ASR_MODEL" ]]; then
    ok "local model directory: $ASR_MODEL"
  elif [[ -f "$HF_MODEL_MARKER" ]]; then
    ok "$ASR_MODEL ($(du -sh "$HF_MODEL_DIR" | cut -f1))"
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    todo "$ASR_MODEL"
  else
    todo "$ASR_MODEL"
    HF_MODEL_ID="$ASR_MODEL" HF_DEST="$HF_MODEL_DIR" HF_REV="$HF_REVISION" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    revision=os.environ["HF_REV"],
    local_dir=os.environ["HF_DEST"],
)
PY
    [[ -f "$HF_MODEL_DIR/config.json" ]] || die "downloaded model has no config.json"
    printf '%s@%s\n' "$ASR_MODEL" "$HF_REVISION" > "$HF_MODEL_MARKER"
    ok "$ASR_MODEL"
  fi
  ;;
*) die "invalid ASR_BACKEND '$ASR_BACKEND' (expected whisper-cpp or huggingface)" ;;
esac

# ---------------------------------------------------------------------------
# 2. Diarization models (GitHub releases)
# ---------------------------------------------------------------------------
echo "Diarization:"
if [[ -f "$SEG_MODEL" ]]; then
  ok "segmentation"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  todo "segmentation"
else
  todo "segmentation"
  T="$WORK/seg.tar.bz2"
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" "$T"
  tar xjf "$T" -C "$MODELS" && rm -f "$T"
  [[ -f "$SEG_MODEL" ]] || die "segmentation model is not where expected"
  ok "segmentation"
fi

if [[ -f "$EMB_MODEL" ]]; then
  ok "speaker embeddings"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  todo "speaker embeddings"
else
  todo "speaker embeddings"
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx" "$EMB_MODEL"
  ok "speaker embeddings"
fi

# ---------------------------------------------------------------------------
# 3. Evaluator LLM: a GGUF from ModelScope, read in place by llama-server
# ---------------------------------------------------------------------------
# No import step: llama-server memory-maps the file where it sits. That is a
# deliberate reason to prefer it over ollama here, which would copy the whole
# ~22 GB into ~/.ollama for no benefit.
echo "Evaluator LLM ($LLM_GGUF_FILE):"
if [[ -f "$LLM_GGUF" ]]; then
  ok "gguf present ($(du -h "$LLM_GGUF" | cut -f1))"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  if [[ -f "$LLM_GGUF.part" ]]; then
    todo "gguf (partial: $(du -h "$LLM_GGUF.part" | cut -f1) downloaded, will resume)"
  else
    todo "gguf"
  fi
else
  todo "gguf"
  fetch "${MODELSCOPE_BASE_URL%/}/models/$LLM_GGUF_REPO/resolve/master/$LLM_GGUF_FILE" "$LLM_GGUF" "GGUF"
  ok "gguf"
fi

if command -v llama-server >/dev/null; then
  # --version writes to stderr and its exit status is not dependable; under
  # `set -e` an unguarded pipeline here would abort the whole script.
  ok "llama-server ($( { llama-server --version 2>&1 || true; } | head -n1))"
else
  todo "llama-server -- install with: brew install llama.cpp"
fi

echo
echo "Done. Next: bin/transcribe.sh"

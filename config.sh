# Shared settings. Sourced by every script in bin/.
# Override any of these in the environment instead of editing, e.g.
#   SPEAKERS=2 bin/transcribe.sh

# --- Speech recognition -----------------------------------------------------
# Backends: whisper-cpp | huggingface. whisper.cpp is the fast, timestamp-aware
# default on Apple silicon. The Hugging Face backend accepts any repository ID
# or local directory compatible with Transformers' automatic-speech-recognition
# pipeline. It transcribes diarized turns so models without timestamp output
# can still be merged with speakers accurately.
: "${ASR_BACKEND:=whisper-cpp}"
: "${ASR_MODEL:=large-v3}"

# Backwards compatibility for existing configurations. Prefer ASR_MODEL now.
if [[ -n "${WHISPER_VARIANT:-}" && "$ASR_MODEL" == "large-v3" ]]; then
  ASR_MODEL="$WHISPER_VARIANT"
fi

# Some Hugging Face repositories, including syvai/hviske-v5.3, ship Python
# model code. Loading that code is explicit because it executes locally.
: "${HF_TRUST_REMOTE_CODE:=0}"
: "${HF_REVISION:=main}"
: "${HF_DEVICE:=auto}"
: "${HF_BATCH_SIZE:=8}"

# Optional quantisation of the ggml model. Empty means keep f16 (~3.1 GB, best
# quality). "q5_0" gives ~1.1 GB and a worthwhile speed-up on M1 for a small
# accuracy cost. Changing this requires re-running bin/fetch-models.sh.
: "${WHISPER_QUANT:=}"

# Spoken language: en | da | auto. A model may support fewer languages.
#
# Deliberately NOT called LANG. LANG is a POSIX locale variable that your shell
# already exports (typically "en_US.UTF-8" or "C.UTF-8"), so a default of
# "${LANG:=en}" would never apply and whisper-cli would be handed a locale
# string as a language code.
: "${AUDIO_LANG:=en}"

# --- Evaluator LLM ----------------------------------------------------------
# Served by llama.cpp's llama-server (Metal). llama.cpp directly rather than
# ollama because ollama is a wrapper over this same engine that would copy the
# whole ~22 GB model into ~/.ollama, and whose 4096-token default window
# silently truncates long transcripts.
#
# Qwen3.6-35B-A3B is a hybrid-attention mixture-of-experts model: 35B total
# parameters but only ~3B active per token. It is substantially newer and more
# capable than Qwen3-30B-A3B while retaining fast sparse decoding on Apple
# silicon. Its native 262k context also gives this 64k configuration ample
# headroom for long transcripts.
#
# This GGUF is the Unsloth conversion mirrored on ModelScope. The UD quantizer
# keeps sensitive tensors at higher precision than a uniform four-bit build.
: "${LLM_GGUF_REPO:=unsloth/Qwen3.6-35B-A3B-GGUF}"
: "${LLM_GGUF_FILE:=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
# ModelScope's international endpoint. The .cn site may be DNS-blocked outside
# China (NextDNS replaces its certificate with a block-page certificate).
: "${MODELSCOPE_BASE_URL:=https://www.modelscope.ai}"

# Context window, passed to llama-server as --ctx-size. A 50-minute
# three-person meeting is roughly 13k tokens. llama-server's own default is
# 4096, which would silently truncate that; see README.
: "${NUM_CTX:=65536}"

# Tokens reserved for the model's own answer. Doubles as the generation cap, so
# the budget check and the real limit agree. Qwen3.6 is a reasoning model and
# spends part of this on a <think> block that gets stripped from the output, so
# it needs more headroom than a non-reasoning model would.
: "${RESERVE_TOKENS:=4000}"

# Where llama-server listens, and how many layers go to the GPU. 999 means
# "all of them", which is what you want on unified memory.
: "${LLM_PORT:=8080}"
: "${LLM_HOST_URL:=http://127.0.0.1:$LLM_PORT}"
: "${LLM_NGL:=999}"

# CPU threads. Empty lets llama.cpp choose. On an M1 Max, pinning this to the 8
# performance cores (LLM_THREADS=8) sometimes beats letting it use the 2
# efficiency cores as well.
: "${LLM_THREADS:=}"

# Seconds to wait for the model to load. A ~22 GB read from cold disk is slow
# the first time and fast afterwards, once it is in the page cache.
: "${LLM_LOAD_TIMEOUT:=600}"

# --- Defaults for a run -----------------------------------------------------
# Fallback speaker count. For recordings with a different number, put that
# number in speakers/<recording name>.count. The per-recording value wins.
: "${SPEAKERS:=3}"

# Re-run a stage even if its output already exists.
: "${FORCE:=0}"

# --- Context ----------------------------------------------------------------
# The markdown file describing the meeting purpose, subject, and criteria. This
# drives evaluation quality more than the model choice does.
#
# CONTEXT_DEFAULT supplies shared criteria. A contexts/<recording name>.md file
# is appended when present. Set CONTEXT_FILE to replace both for every recording.
: "${CONTEXT_DEFAULT:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/contexts/default.md}"
: "${CONTEXT_FILE:=}"

# Sampling for the LLM stages. Low temperature because these jobs are
# extractive: we want fidelity to the transcript, not creative writing.
: "${SUMMARY_TEMPERATURE:=0.2}"
: "${EVALUATION_TEMPERATURE:=0.3}"
: "${COMPARISON_TEMPERATURE:=0.3}"
: "${CHAT_TEMPERATURE:=0.4}"
# Interactive answers can be shorter than the pipeline reports. This reserve
# is checked on every turn so a long conversation cannot silently push the
# source documents out of the model's context window.
: "${CHAT_RESERVE_TOKENS:=2000}"

# --- Speaker naming ---------------------------------------------------------
# bin/name-speakers.sh samples each speaker's longest turns from the opening
# minutes -- the introductions, where people say who they are. Longest rather
# than first because the first turns are "Yeah", "Hi, can you hear me?".
: "${NAME_WINDOW_SECONDS:=420}"
: "${NAME_SAMPLES:=3}"
# Long turns are useful as text but tedious as voice samples. Playback uses the
# opening portion of the matching turn and stops after this many seconds.
: "${NAME_AUDIO_SECONDS:=15}"

# --- Layout -----------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$ROOT/models"
RECORDINGS="$ROOT/recordings"
TRANSCRIPTS="$ROOT/transcripts"
SUMMARIES="$ROOT/summaries"
EVALUATIONS="$ROOT/evaluations"
COMPARISONS="$ROOT/comparisons"
CHATS="$ROOT/chats"
CONTEXTS="$ROOT/contexts"
SPEAKERMAPS="$ROOT/speakers"
WORK="$ROOT/work"
PROMPTS="$ROOT/prompts"
PY="$ROOT/.venv/bin/python"

require_python() {
  [[ -x "$PY" ]] && return 0
  echo "project Python is missing or broken at $PY" >&2
  echo "recreate it with the two uv commands under Setup in README.md" >&2
  return 1
}

GGML_BASE="$MODELS/ggml-$ASR_MODEL.bin"
if [[ -n "$WHISPER_QUANT" ]]; then
  GGML="$MODELS/ggml-$ASR_MODEL-$WHISPER_QUANT.bin"
else
  GGML="$GGML_BASE"
fi
SEG_MODEL="$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL="$MODELS/titanet.onnx"
LLM_GGUF="$MODELS/$LLM_GGUF_FILE"

HF_MODEL_SLUG="${ASR_MODEL//\//--}"
HF_REVISION_SLUG="${HF_REVISION//\//--}"
HF_MODEL_DIR="$MODELS/huggingface/$HF_MODEL_SLUG@$HF_REVISION_SLUG"
HF_MODEL_MARKER="$HF_MODEL_DIR/.complete"

mkdir -p "$MODELS" "$RECORDINGS" "$TRANSCRIPTS" "$SUMMARIES" "$EVALUATIONS" \
         "$COMPARISONS" "$CHATS" "$CONTEXTS" "$SPEAKERMAPS" "$WORK"

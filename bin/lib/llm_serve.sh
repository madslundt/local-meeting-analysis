# llama-server lifecycle. Sourced (not executed) by the stages that need the LLM.
# Assumes config.sh has already been sourced.
#
# The server is started once and left resident so the ~22 GB model loads a
# single time rather than once per recording. bin/llm.sh is the user-facing way
# to check on it or shut it down.

LLM_PID_FILE="$WORK/llama-server.pid"
LLM_STATE_FILE="$WORK/llama-server.state"
LLM_LOG_FILE="$WORK/llama-server.log"

llm_up() { curl -fsS --max-time 3 "$LLM_HOST_URL/health" >/dev/null 2>&1; }

# What the running server was started with, so we can notice a stale one still
# holding the previous model or a smaller context window.
llm_state_now() { printf '%s\t%s\n' "$LLM_GGUF" "$NUM_CTX"; }

llm_stop() {
  if [[ -f "$LLM_PID_FILE" ]]; then
    local pid; pid="$(cat "$LLM_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping llama-server (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$LLM_PID_FILE" "$LLM_STATE_FILE"
    return 0
  fi
  if llm_up; then
    echo "a llama-server is listening on $LLM_HOST_URL but was not started by" >&2
    echo "this project, so it is left alone. Stop it yourself, or set LLM_PORT." >&2
    return 1
  fi
  echo "llama-server is not running"
}

llm_start() {
  command -v llama-server >/dev/null || {
    echo "llama-server not found. Install it with:" >&2
    echo "    brew install llama.cpp" >&2
    return 1
  }
  [[ -f "$LLM_GGUF" ]] || {
    echo "model not found: $LLM_GGUF" >&2
    echo "run bin/fetch-models.sh first" >&2
    return 1
  }

  # Keep the argument array non-empty. macOS ships Bash 3.2, where expanding an
  # empty array under `set -u` fails with "unbound variable".
  local server_args=(
    --model "$LLM_GGUF"
    --ctx-size "$NUM_CTX"
    --n-gpu-layers "$LLM_NGL"
    --host 127.0.0.1 --port "$LLM_PORT"
    --jinja
  )
  [[ -n "$LLM_THREADS" ]] && server_args+=( -t "$LLM_THREADS" )

  echo "starting llama-server: $(basename "$LLM_GGUF"), ctx $NUM_CTX"
  echo "    loading $(du -h "$LLM_GGUF" | cut -f1) -- slow on a cold cache, fast after"
  # --jinja makes llama-server apply the chat template embedded in the GGUF.
  # Without it the server guesses a template, which for a reasoning model like
  # Qwen3.6 produces subtly degraded output rather than an obvious error.
  nohup llama-server "${server_args[@]}" \
    >"$LLM_LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$LLM_PID_FILE"
  llm_state_now > "$LLM_STATE_FILE"

  local waited=0
  while ! llm_up; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "llama-server died while loading. Last lines of $LLM_LOG_FILE:" >&2
      tail -n 20 "$LLM_LOG_FILE" >&2
      rm -f "$LLM_PID_FILE" "$LLM_STATE_FILE"
      return 1
    fi
    if (( waited >= LLM_LOAD_TIMEOUT )); then
      echo "llama-server did not become healthy within ${LLM_LOAD_TIMEOUT}s." >&2
      echo "It may still be loading -- raise LLM_LOAD_TIMEOUT, or watch:" >&2
      echo "    tail -f $LLM_LOG_FILE" >&2
      return 1
    fi
    sleep 2; waited=$((waited + 2))
    (( waited % 20 == 0 )) && echo "    still loading (${waited}s)"
  done
  echo "    ready after ${waited}s"
}

# Idempotent: a healthy server with the right model and window is left alone.
llm_ensure() {
  if llm_up; then
    if [[ -f "$LLM_STATE_FILE" ]] && [[ "$(cat "$LLM_STATE_FILE")" == "$(llm_state_now)" ]]; then
      return 0
    fi
    echo "llama-server is running with a different model or context size; restarting"
    llm_stop || return 1
  fi
  llm_start
}

# Printed at the end of a stage: the model stays in memory on purpose, but 22 GB
# of resident memory is not something to leave behind silently.
llm_note_resident() {
  echo
  echo "llama-server is still running and holding the model in memory."
  echo "Leave it if you are about to run another stage; otherwise: bin/llm.sh stop"
}

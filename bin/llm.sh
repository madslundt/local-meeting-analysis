#!/usr/bin/env bash
# Manage the local LLM server by hand.
#
#   bin/llm.sh start     load the model and leave it resident
#   bin/llm.sh stop      shut it down and free the memory
#   bin/llm.sh status    is it up, and with what
#   bin/llm.sh log       follow the server log
#
# The summarize / evaluate / compare stages start it themselves, so this is for
# reclaiming the ~22 GB when you are done, or for debugging a failed load.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
source "$ROOT/bin/lib/llm_serve.sh"

case "${1:-status}" in
  start)  llm_ensure ;;
  stop)   llm_stop ;;
  status)
    if llm_up; then
      echo "up at $LLM_HOST_URL"
      if [[ -f "$LLM_STATE_FILE" ]]; then
        IFS=$'\t' read -r m c < "$LLM_STATE_FILE"
        echo "  model    $(basename "$m")"
        echo "  ctx-size $c"
        [[ "$m" == "$LLM_GGUF" && "$c" == "$NUM_CTX" ]] \
          || echo "  NOTE: config.sh now wants $(basename "$LLM_GGUF") at ctx $NUM_CTX;"
        [[ "$m" == "$LLM_GGUF" && "$c" == "$NUM_CTX" ]] \
          || echo "        the next stage will restart the server."
      else
        echo "  started outside this project -- no recorded state"
      fi
    else
      echo "down"
      [[ -f "$LLM_GGUF" ]] && echo "  model present: $(basename "$LLM_GGUF")" \
                           || echo "  model missing -- run bin/fetch-models.sh"
    fi
    ;;
  log)    tail -f "$LLM_LOG_FILE" ;;
  *)      echo "usage: bin/llm.sh {start|stop|status|log}" >&2; exit 2 ;;
esac

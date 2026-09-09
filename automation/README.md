# Weekly n8n model monitor

Import `n8n-weekly-model-monitor.json` into n8n. The workflow is intentionally
inactive on import so you can run it manually once and inspect its output before
activating it.

## What it does

- Runs every Monday at 09:00 in the `Europe/Copenhagen` timezone.
- Checks recent releases from official Hugging Face publisher accounts.
- Uses `Qwen/Qwen3.6-35B-A3B` at `UD-Q4_K_M` as the baseline.
- Filters out base checkpoints, quantized duplicates, embeddings, ASR, guard,
  reward, and other specialist models.
- Estimates whether a Q4 build would leave usable headroom on a 64-GB M1 Max.
- Separates likely same-family quality upgrades from cross-family models that
  still need an apples-to-apples benchmark.
- Remembers candidate IDs in n8n workflow static data, so the notification path
  only runs for newly discovered candidates.
- Never downloads a model or modifies `config.sh`.

## Setup

1. In n8n, choose **Import from File** and select the workflow JSON.
2. Run **Run manually** once.
3. Open **Assess Mac fit and upgrade evidence** and inspect `markdown`,
   `candidates`, and `newCandidates` in the output.
4. Connect a Gmail, Slack, Telegram, ntfy, or other notification node after
   **CONNECT NOTIFIER HERE**. Use `{{$json.markdown}}` as the message body.
5. Activate the workflow.

The first active scheduled execution treats all matching current candidates as
new. Later executions notify only when an unseen candidate appears. Manual test
executions may not persist static workflow data, depending on the n8n version;
production scheduled executions do.

## What “better” means here

This is a conservative discovery gate, not an automatic benchmark laboratory.
A model becomes a candidate only when it is newer than the baseline, relevant
to conversational text or multimodal work, plausibly fits the Mac, and has
multiple quality signals such as publisher evaluation results, capacity, MoE
efficiency, or a newer Qwen generation.

Publisher benchmarks are not always directly comparable. Before changing the
pipeline, use identical meeting prompts to compare output quality, generation
speed, memory consumption, Danish/English handling, and long-context fidelity.

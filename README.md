# Meeting Analysis

A fully local pipeline that turns any recorded meeting into:

- timestamped, speaker-labelled transcripts;
- factual summaries;
- optional evaluations against your own criteria; and
- an optional comparison across multiple meetings.

Audio, transcripts, and generated reports stay on this Mac. The pipeline only
needs network access during setup to download its models.

## At a glance

| | |
|---|---|
| **Input** | `.mp3`, `.wav`, `.m4a`, `.mp4`, or `.flac` recordings |
| **Output** | Transcripts, summaries, evaluations, and one comparison report |
| **Processing** | Local models; Metal is used where the selected runtime supports it |
| **Languages** | Any language supported by the selected ASR model; `en`, `da`, and `auto` are common settings |
| **Privacy** | Recordings and derived files are gitignored and are not sent to an API |
| **Configuration** | ASR/LLM models, language, context size, speaker count, and sampling live in `config.sh` |

The pipeline keeps evidence and interpretation separate. A raw transcript is
never overwritten; naming, summarizing, evaluating, and comparing each create a
new file.

## Quick start

### 1. Install dependencies and models

```bash
brew install uv ffmpeg whisper-cpp llama.cpp
uv sync --locked
bin/fetch-models.sh
```

`uv sync --locked` creates `.venv` and installs the exact dependency versions
recorded in `uv.lock`. Run `uv lock --upgrade` when you intentionally want to
update them.

Model downloads are resumable and existing models are skipped. To check what is
missing without downloading anything:

```bash
CHECK_ONLY=1 bin/fetch-models.sh
```

### 2. Add recordings and criteria

Put recordings in `recordings/`. Then create the shared evaluation context:

```bash
cp contexts/example.md contexts/default.md
```

Edit `contexts/default.md` with the meeting purpose, subject to assess, and
criteria shared by the recordings. Skip this if you only need transcripts and
summaries. Evaluation refuses to use the unchanged template.

The default is three English-speaking participants. Override it for a run when
needed:

```bash
SPEAKERS=2 AUDIO_LANG=da bin/transcribe.sh
```

### 3. Run the pipeline

```bash
bin/transcribe.sh       # audio -> anonymous speaker transcript
bin/name-speakers.sh    # identify SPEAKER_0, SPEAKER_1, ...
bin/summarize.sh        # transcript -> factual summary
bin/evaluate.sh         # transcript + criteria -> evaluation
bin/compare.sh          # all evaluations -> comparison
bin/chat.sh             # explore generated documents interactively
```

Run `compare.sh` only when there are at least two evaluations. Every command
skips outputs that are already current. Add `--fresh` to rebuild a stage:

```bash
bin/evaluate.sh --fresh
```

## How it works

```text
recordings/
    |
    |  FFmpeg: normalize audio
    |  pyannote + TitaNet: determine who spoke when
    |  Configured ASR model: determine what was said
    v
transcripts/<name>_transcript.txt
    |
    |  human speaker naming
    v
transcripts/<name>_labelled_transcript.txt
    |                         |
    | Qwen: summarize         | Qwen + context: evaluate
    v                         v
summaries/                evaluations/
                              |
                              | Qwen: compare
                              v
                    comparisons/comparison.md
```

### Workflow and models

| Step | Command | Model or engine | Purpose | Main output |
|---|---|---|---|---|
| Normalize | `bin/transcribe.sh` | FFmpeg | Convert audio to 16 kHz mono PCM | `work/<name>.16k.wav` |
| Diarize | `bin/transcribe.sh` | pyannote segmentation 3.0 + NeMo TitaNet Large | Find speech turns and cluster them by speaker | `work/<name>.diarization.json` |
| Transcribe | `bin/transcribe.sh` | Configured whisper.cpp or Hugging Face ASR model | Produce timestamped text | `work/<name>.asr.json` |
| Merge | `bin/transcribe.sh` | Deterministic time-overlap matching | Attach each text segment to its most-overlapping speaker | `transcripts/<name>_transcript.txt` |
| Name speakers | `bin/name-speakers.sh` | Human review | Replace anonymous clusters with names or roles | `transcripts/<name>_labelled_transcript.txt` |
| Summarize | `bin/summarize.sh` | Qwen3.6-35B-A3B | Create a context-free record of the conversation | `summaries/<name>_summary.md` |
| Evaluate | `bin/evaluate.sh` | Qwen3.6-35B-A3B | Judge the full transcript against your criteria | `evaluations/<name>_evaluation.md` |
| Compare | `bin/compare.sh` | Qwen3.6-35B-A3B | Compare all evaluations using one shared yardstick | `comparisons/comparison.md` |

The summary is intentionally context-free, so it remains useful if the
evaluation criteria change. Evaluation reads the full transcript—not the
summary—so conclusions can retain timestamps and supporting evidence.
Comparison reads the evaluations instead of several full transcripts, which
keeps the prompt within the model's context window.

### Chat with generated documents

Start an interactive, fully local chat over any generated transcripts,
summaries, evaluations, or comparisons:

```bash
bin/chat.sh
bin/chat.sh summaries/meeting_summary.md evaluations/meeting_evaluation.md
```

With no paths, the command shows a numbered document picker. Individual
transcripts can be selected by number; `all` selects every summary, evaluation,
and comparison but excludes transcripts to preserve context space. Ask
follow-up questions, dig into a topic, or request a reformulation; the full
conversation is retained between turns. Multiline text pasted into the prompt
is sent as one question. Use `/sources` to see what is loaded,
`/save [PATH]` to save a summary at any time, and `/quit` to exit. On exit, the
command offers to write a concise Markdown recap to `chats/`.

For scripted use, `--save-summary [PATH]` saves on exit without asking and
`--no-save-prompt` suppresses the exit prompt. Chat summaries, like the other
derived documents, are gitignored.

### Model inventory

| Model | Role | Format and local size | Runtime | Key setting |
|---|---|---:|---|---|
| Whisper `large-v3` | Default multilingual speech recognition | GGML f16, about 2.9 GB | whisper.cpp + Metal | `ASR_MODEL=large-v3`; optional `WHISPER_QUANT=q5_0` |
| Syvai Hviske `v5.3` | Optional Danish speech recognition | Safetensors, about 4.1 GB | Transformers | `ASR_BACKEND=huggingface`; `AUDIO_LANG=da` |
| pyannote segmentation 3.0 | Speech and turn segmentation | ONNX, about 5.7 MB | sherpa-onnx | 0.3 s minimum speech; 0.5 s minimum silence |
| NeMo TitaNet Large | Speaker embeddings | ONNX, about 97 MB | sherpa-onnx | Fixed cluster count from `SPEAKERS` |
| Qwen3.6-35B-A3B UD-Q4_K_M | Summary, evaluation, comparison, and chat | GGUF, about 21 GiB | llama.cpp + Metal | 65,536-token context; 4,000-token report reserve |

Qwen is a mixture-of-experts model with 35B total parameters and about 3B
active per token. The sparse active parameter count provides faster generation
while the larger total capacity supports analysis. `llama-server` memory-maps
the GGUF directly and applies its embedded chat template.

### Danish meetings

The default multilingual Whisper model supports Danish. For a Danish batch,
set the language explicitly; this avoids language detection mistakes on short
openings, names, and English technical terms:

```bash
AUDIO_LANG=da bin/transcribe.sh
```

For a Danish-specialized model, the Hugging Face backend supports
[Syvai's `syvai/hviske-v5.3`](https://huggingface.co/syvai/hviske-v5.3). It is
optimized for Danish conversational speech and ships
custom model code, so enabling that code is an explicit trust decision:

```bash
ASR_BACKEND=huggingface ASR_MODEL=syvai/hviske-v5.3 \
  HF_TRUST_REMOTE_CODE=1 AUDIO_LANG=da bin/fetch-models.sh

ASR_BACKEND=huggingface ASR_MODEL=syvai/hviske-v5.3 \
  HF_TRUST_REMOTE_CODE=1 AUDIO_LANG=da bin/transcribe.sh
```

Hviske is licensed CC BY-NC 4.0; check the model card before commercial use.
Because it does not emit verbose timestamps, the pipeline transcribes each
diarized speaker turn and uses that turn's boundaries as timestamps.

The backend is not limited to Hviske. Set `ASR_MODEL` to any public Hugging Face
repository ID, a gated/private repository available through `HF_TOKEN`, or a
local model directory compatible with the Transformers
`automatic-speech-recognition` pipeline. Standard models normally leave
`HF_TRUST_REMOTE_CODE=0`; enable it only for repositories whose code you trust.
Not every model accepts a language hint, so the runner retries without one when
necessary. Model language coverage, memory use, and Apple-silicon support vary.

## Using each stage

### Speaker naming

Diarization separates voices but does not know their identities, so the first
transcript uses arbitrary labels such as `SPEAKER_0`. The naming command shows:

- the opening exchange;
- each speaker's longest early turns;
- each speaker's share of speaking time; and
- playable excerpts when text alone is not enough (`p1`, `p2`, and so on).

Enter the name or role that should appear in the transcript. Type `n` to reveal
another quote. Giving two clusters the same name merges them, which is useful
when the diarizer splits one person into two voices. Re-run with `--fresh` to
correct a map; previous answers return as defaults. While a clip is playing,
start typing the speaker's name to stop playback immediately and continue the
answer with the key you pressed.

`SPEAKERS=3` is the project default. For one recording with a different count,
put the number in `speakers/<recording-name>.count` before transcribing. That
per-recording value takes precedence over the default.

Speaker names are annotations, not verified ground truth. Generated reports are
prompted to preserve uncertainty when a conclusion depends on attribution.

### Evaluation context

Use `contexts/default.md` for shared criteria. Optionally add
`contexts/<recording-name>.md` for facts specific to one meeting; evaluation
combines the two. Set `CONTEXT_FILE` in the environment or `config.sh` to use
one complete replacement context for every recording.

Write criteria that can be supported or contradicted by the transcript. For
example, “at least one senior reviews my PRs weekly” is testable; “good culture”
is not. Comparison always uses one shared context so every subject is judged by
the same standard.

### Caching and regeneration

Stages rebuild outputs when their transcript, context, or prompt is newer. They
otherwise skip completed work. `--fresh` forces regeneration; `FORCE=1` is the
equivalent environment setting.

Outputs are replaced atomically, so interrupting a rerun leaves the previous
completed file intact. Summary and evaluation prefer the labelled transcript
but fall back to the anonymous one. Comparison refuses to use stale evaluations
and asks you to rerun evaluation first.

### LLM server

The first LLM stage starts `llama-server`. Later stages reuse it to avoid
loading the 21 GiB model again. It remains resident after the pipeline finishes:

```bash
bin/llm.sh status    # show model and context window
bin/llm.sh log       # follow the server log
bin/llm.sh stop      # release its memory
```

Changing `LLM_GGUF_FILE` or `NUM_CTX` causes the next LLM stage to restart the
server with the new configuration.

## Performance

All batch stages have now been run. Across three recordings containing 2:34:48
of audio, the machine completed the measured pipeline work in 1:33:34 (about
0.60x the source duration). This is summed compute time; some stages were run
concurrently, so it is not an end-to-end wall-clock measurement.

### Transcription benchmark (original Whisper setup)

| Recording | Audio | Diarization | Whisper | Compute total | Diarization | Whisper |
|---|---:|---:|---:|---:|---:|---:|
| `2026-09-07_15_05_30.wav` | 54:39 | 22:35 | 5:21 | 27:56 | 80.8% | 19.2% |
| `2026-09-07_16_00_32.wav` | 49:13 | 16:31 | 4:49 | 21:20 | 77.4% | 22.6% |
| `2026-09-08_12_59_48.wav` | 50:56 | 23:13 | 4:30 | 27:43 | 83.8% | 16.2% |
| **Combined** | **2:34:48** | **1:02:19** | **14:40** | **1:16:59** | **80.9%** | **19.1%** |

### Full batch

| Stage | Inputs | Runtime | Share of measured compute |
|---|---:|---:|---:|
| Diarization and Whisper ASR | 3 recordings | 1:16:59 | 82.3% |
| Summarization | 3 transcripts | 10:10 | 10.9% |
| Evaluation | 3 transcripts | 5:04 | 5.4% |
| Comparison | 3 evaluations | 1:21 | 1.4% |
| **Combined** |  | **1:33:34** | **100.0%** |

Normalization and merging were negligible. Manual speaker naming is excluded.
The summarization figure includes the initial Qwen server/model startup; the
later LLM stages reused the resident model.

These elapsed times were reconstructed from shell-history start times and output
modification times from the 8--9 September 2026 run. Treat them as local
observations, not general benchmarks.

## Important constraints

### Context window

A 50-minute, three-person transcript is roughly 13,000 tokens. The configured
window is 65,536 tokens; llama.cpp's 4,096-token default would silently omit
most of a long meeting.

The project passes `NUM_CTX` to the server, estimates prompt size before sending
it, and checks the server's reported token count afterwards. Oversized prompts
are refused instead of truncated. Raise `NUM_CTX` for longer recordings, with
the understanding that a larger context uses more memory. There is currently no
chunking; the exact practical duration depends on speaking density and language.

### Recording quality

The current recordings come from a mono Plaud Note placed on a table. Use the
raw export when possible: aggressive noise reduction can remove voice-timbre
cues needed for speaker clustering. Distance from the recorder, crosstalk, and
three or more speakers all reduce attribution accuracy.

ASR models can hallucinate during long silences. Segments with no overlapping
diarized speech are preserved as `UNATTRIBUTED` rather than silently discarded.
A segment spanning a speaker change is assigned wholly to whichever speaker has
the greatest overlap.

`AUDIO_LANG` applies to an entire run. Process mixed-language batches in
separate passes. The English-trained TitaNet speaker embeddings may need
validation for your language and recording conditions.

## Model downloads and network restrictions

The fetch script uses:

| Asset | Source |
|---|---|
| Whisper checkpoint | OpenAI's Azure Edge CDN, then local GGML conversion |
| Optional Transformers ASR model | Hugging Face |
| Diarization models | sherpa-onnx GitHub releases |
| Qwen GGUF | ModelScope's international endpoint |

Direct binary downloads are checked for expected headers. They use a `.part`
file and are renamed only after completion. Hugging Face snapshots use the hub
client's own cache and integrity handling.

## Privacy and cleanup

Recordings, transcripts, contexts, speaker maps, reports, models, and working
files are gitignored because they can contain sensitive information. Delete them
deliberately when they are no longer needed.

`work/` contains rebuildable intermediates, including roughly 115 MB of PCM
audio per recorded hour. It is safe to clear, but doing so means later runs must
normalize, diarize, and transcribe again.

## Project status

The whisper.cpp pipeline and LLM stages have been exercised on the recordings
behind the benchmark above. The Hugging Face adapter is covered by unit tests;
running a full Hviske transcription still requires downloading its 4.1 GB
checkpoint and validating speed and memory use on the target machine.

## Project layout

```text
config.sh                 shared configuration
bin/fetch-models.sh       resumable model acquisition
bin/transcribe.sh         audio -> raw transcripts
bin/name-speakers.sh      interactive speaker naming
bin/summarize.sh          transcripts -> summaries
bin/evaluate.sh           transcripts + context -> evaluations
bin/compare.sh            evaluations -> comparison
bin/llm.sh                inspect or stop llama-server
bin/lib/                  Python helpers and LLM server lifecycle
prompts/                  editable prompts for LLM stages
contexts/example.md       evaluation-context template
speakers/example.json     speaker-map example
```

To change report structure or emphasis, edit the corresponding files in
`prompts/`. To change models, language, context size, or defaults, edit
`config.sh` or override its variables in the environment.

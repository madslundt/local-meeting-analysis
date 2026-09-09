#!/usr/bin/env python3
"""Transcribe diarized turns with any Transformers-compatible ASR model.

Many speech models return text but no timestamps. Using diarized turns as input
chunks preserves the timing and speaker boundary needed by the merge stage.
Adjacent turns from the same speaker are joined to give the recognizer useful
context, then long turns are split into bounded chunks.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_chunks(turns, max_seconds=30.0, join_gap=0.35):
    chunks = []
    for turn in sorted(turns, key=lambda item: item["start"]):
        start, end = float(turn["start"]), float(turn["end"])
        if end <= start:
            continue
        if (
            chunks
            and chunks[-1]["speaker"] == turn.get("speaker")
            and start - chunks[-1]["end"] <= join_gap
            and end - chunks[-1]["start"] <= max_seconds
        ):
            chunks[-1]["end"] = end
        else:
            chunks.append({"start": start, "end": end, "speaker": turn.get("speaker")})

    bounded = []
    for chunk in chunks:
        start = chunk["start"]
        while start < chunk["end"]:
            end = min(chunk["end"], start + max_seconds)
            bounded.append({**chunk, "start": start, "end": end})
            start = end
    return bounded


def choose_device(requested, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe(pipe, inputs, language, batch_size):
    kwargs = {"batch_size": batch_size}
    if language != "auto":
        kwargs["generate_kwargs"] = {"language": language}
    try:
        result = pipe(inputs, **kwargs)
    except (TypeError, ValueError) as error:
        if "generate_kwargs" not in kwargs:
            raise
        print(
            f"warning: model rejected language={language!r}; retrying without a language hint ({error})",
            file=sys.stderr,
        )
        kwargs.pop("generate_kwargs")
        result = pipe(inputs, **kwargs)
    return result if isinstance(result, list) else [result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        import soundfile as sf
        import torch
        from transformers import pipeline
    except ImportError as error:
        sys.exit(f"Hugging Face ASR dependencies are missing: {error}\nRun: uv sync --locked")

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) != 1:
        audio = audio.mean(axis=1)
    diarization = json.loads(args.diarization.read_text())
    chunks = build_chunks(diarization.get("turns", []))
    if not chunks:
        sys.exit("diarization contains no usable speech turns")

    device = choose_device(args.device, torch)
    print(f"   loading {args.model_id} on {device}", file=sys.stderr)
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device=device,
    )
    inputs = []
    for chunk in chunks:
        start = max(0, round(chunk["start"] * sample_rate))
        end = min(len(audio), round(chunk["end"] * sample_rate))
        inputs.append({"raw": audio[start:end], "sampling_rate": sample_rate})

    results = transcribe(recognizer, inputs, args.language, args.batch_size)
    if len(results) != len(chunks):
        sys.exit(f"ASR returned {len(results)} results for {len(chunks)} chunks")

    transcription = []
    for chunk, result in zip(chunks, results):
        text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        if text:
            transcription.append({
                "text": text,
                "offsets": {
                    "from": round(chunk["start"] * 1000),
                    "to": round(chunk["end"] * 1000),
                },
            })

    payload = {
        "backend": "huggingface",
        "model": args.model_id,
        "language": args.language,
        "transcription": transcription,
    }
    part = args.output.with_suffix(args.output.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(part, args.output)


if __name__ == "__main__":
    main()

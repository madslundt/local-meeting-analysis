#!/usr/bin/env python3
"""Speaker diarization via sherpa-onnx. Writes JSON turns to stdout.

Deliberately takes the speaker count as a required argument rather than
estimating it: we always know how many people were in the room, and telling
the clusterer beats making it guess.
"""

import argparse
import json
import sys
import wave

import numpy as np
import sherpa_onnx

MODELS = None  # set from args


def read_wav_mono_f32(path, expected_rate):
    """Read a 16-bit PCM mono wav as float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            sys.exit(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        if w.getframerate() != expected_rate:
            sys.exit(f"{path}: expected {expected_rate} Hz, got {w.getframerate()} Hz")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build(seg_model, emb_model, num_speakers):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        sys.exit("sherpa-onnx rejected the diarization config; check model paths")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="16 kHz mono 16-bit PCM wav")
    ap.add_argument("--segmentation", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--speakers", type=int, required=True)
    args = ap.parse_args()

    sd = build(args.segmentation, args.embedding, args.speakers)
    samples = read_wav_mono_f32(args.wav, sd.sample_rate)

    def progress(processed, total):
        pct = 100.0 * processed / total if total else 0.0
        print(f"\rdiarizing: {pct:5.1f}%", end="", file=sys.stderr, flush=True)
        return 0

    result = sd.process(samples, callback=progress).sort_by_start_time()
    print(file=sys.stderr)

    turns = [
        {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
        for s in result
    ]
    json.dump({"speakers": args.speakers, "turns": turns}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

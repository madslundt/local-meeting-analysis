#!/usr/bin/env python3
"""Attach speaker labels to ASR segments and emit a readable transcript.

Each ASR segment is assigned the speaker whose diarized turns overlap it
most in time. Segments are then coalesced into consecutive same-speaker blocks.

Known limitation: a single ASR segment that straddles a speaker change is
attributed wholly to one speaker. Most ASR segments end on natural pauses, which
usually coincide with turn boundaries, so this is mostly benign -- but in
crosstalk it will misattribute a clause. Reported at the end as a confidence
figure so you can judge whether to trust the labels.

Also imported by name_speakers.py, which needs the same segment loading and
overlap assignment.
"""

import argparse
import json
import sys
from collections import defaultdict


def hhmmss(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def load_asr_segments(path):
    """Pull (start, end, text) from the pipeline's canonical ASR JSON."""
    with open(path) as f:
        data = json.load(f)
    out = []
    for seg in data.get("transcription", []):
        off = seg.get("offsets") or {}
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append((off["from"] / 1000.0, off["to"] / 1000.0, text))
    return out


# Backwards compatible internal alias.
load_whisper_segments = load_asr_segments


def load_turns(path):
    with open(path) as f:
        return json.load(f)["turns"]


def load_names(path):
    """Speaker-id -> display name, from a speakers/<name>.json map."""
    if not path:
        return {}
    with open(path) as f:
        return {str(k): v for k, v in json.load(f).items()}


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign(segments, turns):
    """Give every ASR segment the speaker who overlaps it most.

    Returns dicts with speaker (None if no diarized speech overlaps), start,
    end, text and conf -- the share of overlapping speech held by the winner.
    """
    labelled = []
    for s0, s1, text in segments:
        totals = defaultdict(float)
        for t in turns:
            ov = overlap(s0, s1, t["start"], t["end"])
            if ov > 0:
                totals[t["speaker"]] += ov
        if not totals:
            # Diarizer found no speech here -- usually an ASR hallucination
            # over silence. Keep it, flagged, rather than silently dropping it.
            labelled.append({"speaker": None, "start": s0, "end": s1,
                             "text": text, "conf": 0.0})
            continue
        speaker = max(totals, key=totals.get)
        claimed = sum(totals.values())
        labelled.append({
            "speaker": speaker, "start": s0, "end": s1, "text": text,
            "conf": totals[speaker] / claimed if claimed else 0.0,
        })
    return labelled


def resolve_label(speaker, names):
    if speaker is None:
        return "UNATTRIBUTED"
    return names.get(str(speaker), f"SPEAKER_{speaker}")


def coalesce(labelled, names):
    """Merge runs of same-label segments into single blocks.

    Grouping on the resolved label rather than the speaker id matters when two
    diarized speakers were given the same name -- which is exactly how you fix
    a recording where one person got split across two labels.
    """
    blocks = []
    for seg in labelled:
        label = resolve_label(seg["speaker"], names)
        if blocks and blocks[-1]["label"] == label:
            blocks[-1]["text"] += " " + seg["text"]
            blocks[-1]["end"] = seg["end"]
        else:
            blocks.append({"label": label, "start": seg["start"],
                           "end": seg["end"], "text": seg["text"]})
    return blocks


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--asr-json")
    source.add_argument("--whisper-json", help=argparse.SUPPRESS)
    ap.add_argument("--diarization-json", required=True)
    ap.add_argument("--speakers-map",
                    help='JSON mapping e.g. {"0": "Amina (facilitator)"}')
    args = ap.parse_args()

    segments = load_asr_segments(args.asr_json or args.whisper_json)
    if not segments:
        sys.exit("no ASR segments found -- did transcription produce output?")
    turns = load_turns(args.diarization_json)
    names = load_names(args.speakers_map)

    labelled = assign(segments, turns)
    blocks = coalesce(labelled, names)

    for b in blocks:
        print(f"[{hhmmss(b['start'])}] {b['label']}: {b['text'].strip()}")

    confs = [s["conf"] for s in labelled if s["conf"] > 0]
    mean = sum(confs) / len(confs) if confs else 0.0
    unattributed = sum(1 for s in labelled if s["speaker"] is None)
    print(
        f"\n-- {len(blocks)} turns, {len(segments)} segments, "
        f"mean attribution confidence {mean:.0%}, "
        f"{unattributed} unattributed segment(s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

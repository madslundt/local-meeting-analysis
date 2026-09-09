#!/usr/bin/env python3
"""Ask who each diarized speaker is, then write a speakers/<name>.json map.

Diarization tells voices apart but has no idea whose they are, so it emits
SPEAKER_0/1/2 in arbitrary order. This shows you enough of each voice to
recognise it and writes down your answers.

What it shows, and why:

- The opening exchange in order. Introductions are a sequence -- "I'm Anna, I
  lead the team", "and I'm Tom, I'd be your closest colleague" -- and reading
  them in order is usually all it takes.
- Then each speaker's LONGEST turns from the opening minutes, not their first.
  First turns are "Yeah", "Hi, can you hear me?", "Sorry, one second" and
  identify nobody.
- Each speaker's share of total speaking time, which is a useful cross-check:
  in many meetings the facilitator is not the person talking most.

Several turns per speaker rather than one, because that is how you catch the
failure case where one person was split across two labels. Give two labels the
same name and they merge.
"""

import argparse
import json
import select
import shutil
import subprocess
import sys
import termios
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge import assign, hhmmss, load_asr_segments, load_turns  # noqa: E402

MAX_QUOTE_CHARS = 320
DEFAULT_MAX_AUDIO_SECONDS = 15.0
PLAYBACK_POLL_SECONDS = 0.05


def truncate(text):
    text = " ".join(text.split())
    return text if len(text) <= MAX_QUOTE_CHARS else text[:MAX_QUOTE_CHARS - 1] + "…"


def speaking_time(turns):
    totals = defaultdict(float)
    for t in turns:
        totals[str(t["speaker"])] += t["end"] - t["start"]
    return totals


def opening_exchange(labelled, limit):
    """The first `limit` speaker changes, in order, as (label, time, text)."""
    out = []
    for seg in labelled:
        if seg["speaker"] is None:
            continue
        key = str(seg["speaker"])
        if out and out[-1][0] == key:
            out[-1][2] += " " + seg["text"]
            continue
        if len(out) >= limit:
            break
        out.append([key, seg["start"], seg["text"]])
    return out


def samples_for(labelled, speaker, window, count):
    """That speaker's longest turns inside the opening window.

    Falls back to their longest anywhere if they said nothing substantial early
    -- someone who only speaks up after half an hour still needs a name.
    """
    mine = [s for s in labelled if str(s["speaker"]) == speaker]
    early = [s for s in mine if s["start"] <= window]
    pool, late = (early, False) if early else (mine, True)
    pool = sorted(pool, key=lambda s: len(s["text"]), reverse=True)[:count]
    return sorted(pool, key=lambda s: s["start"]), late


def remaining_samples(labelled, speaker, window, shown):
    """Return more useful quotes for a speaker, best samples first.

    Prefer the opening window, then continue through later speech. `shown`
    contains the initial samples and must not be offered again.
    """
    mine = [s for s in labelled if str(s["speaker"]) == speaker]
    shown_ids = {id(s) for s in shown}
    unseen = [s for s in mine if id(s) not in shown_ids]
    return sorted(
        unseen,
        key=lambda s: (s["start"] > window, -len(s["text"]), s["start"]),
    )


def print_sample(sample, index, numbered):
    prefix = f"[{index}] " if numbered else ""
    print(f"  {prefix}[{hhmmss(sample['start'])}] {truncate(sample['text'])}")
    print()


def stop_process(process):
    """Stop a player promptly, escalating only if it ignores termination."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def play_until_keypress(command):
    """Run a player until it exits or the user presses a key.

    The terminal is temporarily non-canonical so a key is observable without
    waiting for Enter. Echo is disabled while listening; the unread key stays
    queued for the next normal input prompt.
    """
    process = subprocess.Popen(command)
    if not sys.stdin.isatty():
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return None

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    interrupted = False
    try:
        immediate = termios.tcgetattr(fd)
        immediate[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(fd, termios.TCSADRAIN, immediate)

        while process.poll() is None:
            readable, _, _ = select.select(
                [sys.stdin], [], [], PLAYBACK_POLL_SECONDS
            )
            if readable:
                # Leave the key in the terminal's input queue. Once canonical
                # mode is restored, the normal editable input prompt below
                # receives it as the first character of the answer.
                interrupted = True
                stop_process(process)
                break
    except BaseException:
        stop_process(process)
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)

    returncode = process.wait()
    if not interrupted and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return interrupted


def play_sample(audio, sample, max_seconds):
    """Play one timestamped excerpt; report whether a key interrupted it."""
    start = max(0.0, sample["start"] - 0.2)
    duration = min(max_seconds, max(0.1, sample["end"] - start + 0.2))

    # macOS's afplay cannot seek, so first extract a small temporary clip.
    if shutil.which("afplay"):
        with tempfile.TemporaryDirectory(prefix="speaker-sample-") as tmp:
            clip = Path(tmp) / "sample.wav"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-ss", str(start), "-i", str(audio), "-t", str(duration),
                 "-ac", "1", "-c:a", "pcm_s16le", str(clip)],
                check=True,
            )
            return play_until_keypress(["afplay", str(clip)])

    if shutil.which("ffplay"):
        return play_until_keypress(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
             "-ss", str(start), "-t", str(duration), str(audio)]
        )

    raise RuntimeError("no audio player found (expected afplay or ffplay)")


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--asr-json")
    source.add_argument("--whisper-json", help=argparse.SUPPRESS)
    ap.add_argument("--diarization-json", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio", type=Path,
                    help="recording to play while identifying speakers")
    ap.add_argument("--window", type=float, default=420.0,
                    help="seconds of the opening to sample from")
    ap.add_argument("--samples", type=int, default=3,
                    help="turns to show per speaker")
    ap.add_argument("--label", default="", help="recording name, for the header")
    ap.add_argument("--max-audio-seconds", type=float,
                    default=DEFAULT_MAX_AUDIO_SECONDS,
                    help="maximum length of a played sample")
    args = ap.parse_args()

    if args.audio and not args.audio.is_file():
        sys.exit(f"audio file not found: {args.audio}")
    if args.max_audio_seconds <= 0:
        sys.exit("--max-audio-seconds must be greater than zero")

    if not sys.stdin.isatty():
        sys.exit(
            "name_speakers.py needs a terminal to ask you questions.\n"
            "Run bin/name-speakers.sh directly rather than in a pipeline."
        )

    segments = load_asr_segments(args.asr_json or args.whisper_json)
    if not segments:
        sys.exit("no ASR segments found -- did transcription produce output?")
    turns = load_turns(args.diarization_json)
    labelled = assign(segments, turns)

    times = speaking_time(turns)
    total = sum(times.values()) or 1.0
    speakers = sorted(times, key=lambda k: int(k) if k.isdigit() else k)
    if not speakers:
        sys.exit("diarization found no speakers at all -- check work/*.diarization.json")

    existing = {}
    if args.out.is_file():
        existing = {str(k): v for k, v in json.loads(args.out.read_text()).items()}

    print()
    print("=" * 72)
    print(f"  {args.label or args.out.stem}: {len(speakers)} speaker(s) found")
    print("=" * 72)

    print("\nHow it opened:\n")
    for key, start, text in opening_exchange(labelled, limit=8):
        print(f"  [{hhmmss(start)}] SPEAKER_{key}: {truncate(text)}")

    print("\nSpeaking time:\n")
    for key in speakers:
        print(f"  SPEAKER_{key}  {times[key] / 60:5.1f} min  "
              f"({times[key] / total:4.0%} of the talking)")

    print("\nNow the longest thing each of them said early on.")
    print("Enter the name or role that should appear in the transcript.")
    print("Press Enter alone to leave a voice as SPEAKER_n.")
    print("Give two speakers the SAME name to merge them into one person.")

    names = {}
    for key in speakers:
        print()
        print("-" * 72)
        print(f"SPEAKER_{key} -- {times[key] / 60:.1f} min, "
              f"{times[key] / total:.0%} of the talking")
        picks, late = samples_for(labelled, key, args.window, args.samples)
        if late:
            print(f"(nothing in the first {args.window / 60:.0f} min; "
                  f"showing their longest turns from later)")
        print()
        for index, s in enumerate(picks, start=1):
            print_sample(s, index, numbered=bool(args.audio))
        catalog = picks + remaining_samples(labelled, key, args.window, picks)
        shown = set(range(1, len(picks) + 1))
        prior = existing.get(key)
        prompt = f"Who is SPEAKER_{key}?"
        prompt += f" [{prior}] " if prior else " "
        commands = []
        if args.audio and catalog:
            commands.append(f"p1-p{len(catalog)} plays any clip")
        if len(shown) < len(catalog):
            commands.append("n shows another quote")
        if commands:
            print("Type " + "; ".join(commands) + "; or enter the name.")
        while True:
            try:
                answer = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit("\naborted -- nothing written")

            is_play_command = (
                args.audio and len(answer) > 1
                and answer[0].casefold() == "p" and answer[1:].isdigit()
            )
            if is_play_command:
                sample_number = int(answer[1:])
                if not 1 <= sample_number <= len(catalog):
                    print(f"There are only {len(catalog)} clips for this speaker "
                          f"(p1 through p{len(catalog)}).")
                    continue
                sample = catalog[sample_number - 1]
                if sample_number not in shown:
                    print()
                    print_sample(sample, sample_number, numbered=True)
                    shown.add(sample_number)
                print(f"Playing clip {sample_number}… Start typing to stop it.")
                try:
                    if play_sample(args.audio, sample, args.max_audio_seconds):
                        print("Playback stopped; continue typing your answer.")
                except KeyboardInterrupt:
                    print("\nPlayback stopped.")
                except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                    print(f"Could not play the clip: {exc}", file=sys.stderr)
                continue
            if answer.casefold() == "n":
                unseen = [i for i in range(1, len(catalog) + 1) if i not in shown]
                if not unseen:
                    print("No more quotes found for this speaker.")
                    continue
                sample_number = unseen[0]
                sample = catalog[sample_number - 1]
                shown.add(sample_number)
                print()
                print_sample(sample, sample_number, numbered=bool(args.audio))
                if args.audio:
                    print(f"Type p{sample_number} to play this clip.")
                if len(shown) == len(catalog):
                    print("This is the last quote found for this speaker.")
                continue
            break
        if not answer:
            answer = prior or ""
        if answer:
            names[key] = answer

    if not names:
        sys.exit("no names given -- nothing written")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_part = args.out.with_name(args.out.name + ".part")
    out_part.write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n")
    out_part.replace(args.out)

    merged = [n for n in set(names.values()) if list(names.values()).count(n) > 1]
    print(f"\nwrote {args.out}")
    for key in speakers:
        print(f"  SPEAKER_{key} -> {names.get(key, f'SPEAKER_{key} (unnamed)')}")
    if merged:
        print(f"  merging into one person each: {', '.join(sorted(merged))}")


if __name__ == "__main__":
    main()

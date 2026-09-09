#!/usr/bin/env python3
"""Run one prompt template against the local LLM. Nothing leaves the box.

Shared by the summarize, evaluate and compare stages; they differ only in which
prompt files they point at and which fields they supply. Placeholders available
to a template are the ones passed on the command line: {transcript}, {context},
{evaluations}.

Talks to llama.cpp's llama-server over its OpenAI-compatible endpoint. See
bin/lib/llm_serve.sh for the server's lifecycle.

The prompt is measured against the context window before anything is sent.
llama-server's default window is 4096 tokens; a 50-minute three-person
meeting is roughly 13k. Overflowing it does not raise an error -- the model
just silently sees the tail of the conversation and writes a confident,
badly-informed answer. That is the worst failure mode this pipeline has, so it
is turned into a loud refusal here.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Fields a template may reference. Each is a CLI flag of the same name and, if
# given, becomes a placeholder the template can use.
FIELDS = ("transcript", "context", "evaluations")


def estimate_tokens(text):
    """Chars-per-token heuristic, used before the server has seen the prompt.

    3.4 is deliberately pessimistic for English (real is nearer 4.0) and about
    right for Danish, which fragments more. Erring low on chars-per-token means
    erring high on the token count, which fails safe.
    """
    return int(len(text) / 3.4)


def read_field(name, path):
    if not path.is_file():
        if name == "context":
            sys.exit(
                f"context file not found: {path}\n"
                "Set CONTEXT_FILE in config.sh, or add contexts/<recording>.md"
            )
        sys.exit(f"missing {name} file: {path}")
    text = path.read_text()
    if name == "context" and "REPLACE ME" in text:
        sys.exit(
            f"{path} still has REPLACE ME placeholders in it.\n"
            "The evaluation is only as good as this file -- fill it in first."
        )
    return text.strip()


def stream_completion(host, body):
    """Yield content pieces from a streamed chat completion; return usage dict.

    llama-server speaks server-sent events here: each line is 'data: {json}',
    terminated by 'data: [DONE]'. Usage totals arrive in a final chunk that
    carries no choices.
    """
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    usage, finish, saw_reasoning = {}, None, False
    try:
        resp = urllib.request.urlopen(req, timeout=None)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        sys.exit(f"llama-server returned HTTP {e.code}:\n{detail}")
    except urllib.error.URLError as e:
        sys.exit(
            f"cannot reach llama-server at {host}: {e}\n"
            "Start it with: bin/llm.sh start"
        )

    with resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "error" in ev:
                msg = ev["error"]
                sys.exit(f"llama-server error: {msg.get('message', msg) if isinstance(msg, dict) else msg}")
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    saw_reasoning = True
                piece = delta.get("content") or ""
                if piece:
                    yield piece
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    # Attached to the generator so the caller can read them once it is exhausted.
    stream_completion.usage = usage
    stream_completion.finish_reason = finish
    stream_completion.saw_reasoning = saw_reasoning


def completion_body(system, prompt, temperature, reserve):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        # These pipeline stages need the answer, not a long private scratchpad.
        # Qwen's chat template otherwise enables thinking by default and can use
        # the entire generation cap before emitting any answer content.
        "chat_template_kwargs": {"enable_thinking": False},
        # Doubles as the generation cap so the budget check above and the real
        # limit cannot disagree.
        "max_tokens": reserve,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def main():
    ap = argparse.ArgumentParser()
    for name in FIELDS:
        ap.add_argument(f"--{name}", type=Path, help=f"file supplying {{{name}}}")
    ap.add_argument("--system", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-ctx", type=int, default=int(os.environ.get("NUM_CTX", 65536)))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--reserve", type=int, default=int(os.environ.get("RESERVE_TOKENS", 4000)))
    ap.add_argument("--host", default=os.environ.get("LLM_HOST_URL", "http://127.0.0.1:8080"))
    args = ap.parse_args()

    for p in (args.system, args.template):
        if not p.is_file():
            sys.exit(f"missing file: {p}")

    fields = {}
    for name in FIELDS:
        path = getattr(args, name)
        if path is not None:
            fields[name] = read_field(name, path)
    if not fields:
        sys.exit("no input given -- pass at least one of " + ", ".join(f"--{f}" for f in FIELDS))

    template = args.template.read_text()
    try:
        prompt = template.format(**fields)
    except KeyError as e:
        sys.exit(
            f"{args.template} references {{{e.args[0]}}}, which this stage does "
            f"not provide (available: {', '.join(sorted(fields))})"
        )
    system = args.system.read_text().strip()

    est = estimate_tokens(system + prompt)
    if est + args.reserve > args.num_ctx:
        need = 1 << (est + args.reserve).bit_length()
        sys.exit(
            f"prompt is ~{est} tokens and {args.reserve} are reserved for the "
            f"answer, which exceeds the context window of {args.num_ctx}.\n"
            f"Set NUM_CTX={need} in config.sh; the next stage will restart "
            f"llama-server with the larger window.\n"
            f"Refusing to send a prompt that would be silently truncated."
        )
    print(f"    ~{est} prompt tokens / {args.num_ctx} context", file=sys.stderr)

    body = completion_body(system, prompt, args.temperature, args.reserve)

    chunks = []
    for piece in stream_completion(args.host, body):
        chunks.append(piece)
        sys.stderr.write(piece)
        sys.stderr.flush()
    usage = getattr(stream_completion, "usage", {}) or {}
    finish = getattr(stream_completion, "finish_reason", None)

    text = "".join(chunks)
    # Older server/model combinations put scratchpads in a <think> block rather
    # than the API's separate reasoning_content field. Support both formats.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        sys.exit(
            "\nthe model used its whole token budget reasoning and never wrote "
            f"an answer.\nRaise RESERVE_TOKENS in config.sh (currently {args.reserve})."
        )
    text = text.strip()
    if not text:
        if getattr(stream_completion, "saw_reasoning", False):
            sys.exit(
                "\nthe model used its whole token budget reasoning and never "
                "wrote an answer. The server may have ignored "
                "chat_template_kwargs.enable_thinking=false."
            )
        sys.exit("\nmodel returned nothing usable")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the previous completed result if this write is interrupted.
    out_part = args.out.with_name(args.out.name + ".part")
    out_part.write_text(text + "\n")
    out_part.replace(args.out)

    if finish == "length":
        print(
            f"\n    WARNING: output was cut off at the {args.reserve}-token cap. "
            "Raise RESERVE_TOKENS in config.sh and re-run with --fresh.",
            file=sys.stderr,
        )

    # prompt_tokens is the real count from the real tokenizer. If it lands
    # suspiciously close to 4096 the larger window never actually took effect.
    real = usage.get("prompt_tokens")
    if real:
        print(f"\n    server counted {real} prompt tokens", file=sys.stderr)
        if 4000 <= real <= 4096:
            print(
                "    WARNING: that is suspiciously close to llama-server's 4096 "
                "default. The larger window may not have applied; see README.",
                file=sys.stderr,
            )
    print(f"    wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

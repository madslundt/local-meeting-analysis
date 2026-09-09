#!/usr/bin/env python3
"""Chat locally with generated meeting documents and optionally save a recap."""

import argparse
import os
import select
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_run import estimate_tokens, stream_completion


ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIRS = ("summaries", "evaluations", "comparisons")
EXIT_COMMANDS = {"/q", "/quit", "/exit"}
PASTE_GRACE_SECONDS = 0.1


def discover_documents(root=ROOT):
    """Return preferred transcripts and reports in a stable, readable order."""
    documents = []
    transcript_dir = root / "transcripts"
    if transcript_dir.is_dir():
        for raw in sorted(transcript_dir.glob("*_transcript.txt")):
            if raw.name.endswith("_labelled_transcript.txt"):
                continue
            labelled = raw.with_name(raw.name.replace("_transcript.txt", "_labelled_transcript.txt"))
            documents.append(labelled if labelled.is_file() else raw)
    for dirname in GENERATED_DIRS:
        directory = root / dirname
        if directory.is_dir():
            documents.extend(sorted(directory.glob("*.md")))
    return documents


def documents_for_all(documents):
    """The picker shorthand excludes large primary transcripts by design."""
    return [path for path in documents if path.parent.name != "transcripts"]


def choose_documents(documents, input_fn=input, output=sys.stdout):
    if not documents:
        raise SystemExit(
            "No generated documents found. Run bin/transcribe.sh, "
            "bin/summarize.sh, bin/evaluate.sh, or bin/compare.sh first."
        )
    if not sys.stdin.isatty():
        raise SystemExit("No documents specified. Pass one or more paths to bin/chat.sh.")

    print("Available documents:", file=output)
    for number, path in enumerate(documents, 1):
        print(f"  {number:>2}. {path.relative_to(ROOT)}", file=output)
    while True:
        answer = input_fn("Choose numbers (comma-separated), or 'all' for all except transcripts: ").strip()
        if answer.lower() == "all":
            selected = documents_for_all(documents)
            if selected:
                return selected
            print("There are no summaries, evaluations, or comparisons to select.", file=output)
            continue
        try:
            indexes = [int(part.strip()) for part in answer.split(",")]
            if not indexes or any(i < 1 or i > len(documents) for i in indexes):
                raise ValueError
        except ValueError:
            print("Enter document numbers such as 1,3, or 'all'.", file=output)
            continue
        # Preserve order while silently de-duplicating repeated selections.
        return [documents[i - 1] for i in dict.fromkeys(indexes)]


def resolve_documents(values):
    paths = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Document not found: {path}")
        paths.append(path)
    return paths


def read_question(prompt="\nyou> ", input_fn=None):
    """Read one typed line or coalesce the lines from a terminal paste.

    Canonical terminals expose each pasted newline as a separate input() call.
    Once the first line arrives, the rest of a local paste is already readable;
    a short grace period lets us collect it without requiring a special mode.
    Non-interactive stdin keeps its line-at-a-time behavior for scripts.
    """
    input_fn = input_fn or input
    first = input_fn(prompt)
    if not sys.stdin.isatty():
        return first.strip()

    lines = [first]
    while True:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], PASTE_GRACE_SECONDS)
        except (OSError, TypeError, ValueError):
            break
        if not readable:
            break
        try:
            lines.append(input_fn(""))
        except EOFError:
            break

    text = "\n".join(lines)
    # Some terminals expose bracketed-paste control markers to applications
    # that do not use readline. They are transport metadata, not user text.
    return text.replace("\x1b[200~", "").replace("\x1b[201~", "").strip()


def document_message(paths):
    sections = []
    for path in paths:
        text = path.read_text(errors="replace").strip()
        sections.append(f"<document name={path.name!r}>\n{text}\n</document>")
    return "Here are the source documents for this chat:\n\n" + "\n\n".join(sections)


def completion_body(messages, temperature, reserve):
    return {
        "messages": messages,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": reserve,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def message_token_estimate(messages):
    # Allow a little overhead for role markers inserted by the chat template.
    return estimate_tokens("\n".join(m["role"] + ": " + m["content"] for m in messages)) + 16 * len(messages)


def request(messages, host, temperature, reserve, num_ctx, output=sys.stdout):
    estimate = message_token_estimate(messages)
    if estimate + reserve > num_ctx:
        raise ValueError(
            f"This turn needs about {estimate + reserve:,} tokens but NUM_CTX is "
            f"{num_ctx:,}. Start a new chat with fewer documents, or raise NUM_CTX."
        )

    pieces = []
    for piece in stream_completion(host, completion_body(messages, temperature, reserve)):
        pieces.append(piece)
        output.write(piece)
        output.flush()
    text = "".join(pieces).strip()
    if not text:
        raise RuntimeError("The model returned no answer.")
    if getattr(stream_completion, "finish_reason", None) == "length":
        print(
            f"\n[Answer reached the {reserve}-token limit and may be incomplete.]",
            file=sys.stderr,
        )
    return text


def safe_stem(text):
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
    return "-".join(part for part in cleaned.split("-") if part)[:50] or "documents"


def default_summary_path(paths, now=None):
    now = now or datetime.now()
    subject = safe_stem(paths[0].stem)
    if len(paths) > 1:
        subject += f"-and-{len(paths) - 1}-more"
    return ROOT / "chats" / f"{now:%Y-%m-%d_%H%M%S}_{subject}_chat-summary.md"


def save_summary(paths, exchanges, destination, host, temperature, reserve, num_ctx):
    transcript = []
    for question, answer in exchanges:
        transcript.extend((f"## User\n\n{question}", f"## Assistant\n\n{answer}"))
    source_names = ", ".join(path.name for path in paths)
    messages = [
        {"role": "system", "content": (ROOT / "prompts/chat-summary.system.md").read_text().strip()},
        {
            "role": "user",
            "content": f"Source document names: {source_names}\n\n" + "\n\n".join(transcript),
        },
    ]
    print("Generating chat summary…", file=sys.stderr)
    # Summary output should not be echoed into the finished interactive chat.
    from io import StringIO

    summary = request(messages, host, temperature, reserve, num_ctx, output=StringIO())
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    front_matter = (
        f"<!-- Source documents: {source_names} -->\n"
        f"<!-- Saved: {datetime.now().astimezone().isoformat(timespec='seconds')} -->\n\n"
    )
    part.write_text(front_matter + summary + "\n")
    part.replace(destination)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Chat locally with generated transcripts, summaries, evaluations, and comparisons.",
        epilog=(
            "Inside the chat: /sources lists loaded files, /save [PATH] saves a "
            "summary, and /quit exits."
        ),
    )
    parser.add_argument("documents", nargs="*", help="documents to load; omit to choose interactively")
    parser.add_argument(
        "--save-summary",
        nargs="?",
        const="",
        metavar="PATH",
        help="save a summary on exit, optionally at PATH, without asking",
    )
    parser.add_argument("--no-save-prompt", action="store_true", help="do not offer to save on exit")
    parser.add_argument("--host", default=os.environ.get("LLM_HOST_URL", "http://127.0.0.1:8080"), help=argparse.SUPPRESS)
    parser.add_argument("--num-ctx", type=int, default=int(os.environ.get("NUM_CTX", 65536)), help=argparse.SUPPRESS)
    parser.add_argument("--reserve", type=int, default=int(os.environ.get("CHAT_RESERVE_TOKENS", 2000)), help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("CHAT_TEMPERATURE", 0.4)), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = resolve_documents(args.documents) if args.documents else choose_documents(discover_documents())
    system = (ROOT / "prompts/chat.system.md").read_text().strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": document_message(paths)},
        {"role": "assistant", "content": "I have loaded the documents and will use them as the factual basis for our conversation."},
    ]
    exchanges = []
    last_saved_count = 0

    initial_estimate = message_token_estimate(messages)
    if initial_estimate + args.reserve > args.num_ctx:
        raise SystemExit(
            f"The selected documents need about {initial_estimate + args.reserve:,} "
            f"tokens but NUM_CTX is {args.num_ctx:,}. Select fewer documents or raise NUM_CTX."
        )

    print("\nLoaded:")
    for path in paths:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"  - {shown}")
    print("\nAsk about the documents. Commands: /sources, /save [PATH], /quit")

    while True:
        try:
            question = read_question()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        if question.lower() == "/sources":
            for path in paths:
                print(path)
            continue
        if question.lower() == "/help":
            print("/sources  list source files\n/save [PATH]  summarize this chat\n/quit  exit")
            continue
        if question.lower() == "/save" or question.lower().startswith("/save "):
            if not exchanges:
                print("There is no conversation to summarize yet.")
                continue
            supplied = question[5:].strip()
            destination = Path(supplied).expanduser().resolve() if supplied else default_summary_path(paths)
            try:
                saved = save_summary(paths, exchanges, destination, args.host, args.temperature, args.reserve, args.num_ctx)
                print(f"Saved {saved}")
                last_saved_count = len(exchanges)
            except (ValueError, RuntimeError) as error:
                print(f"Could not save summary: {error}", file=sys.stderr)
            continue
        if question.startswith("/"):
            print("Unknown command. Use /help to see available commands.")
            continue

        candidate = messages + [{"role": "user", "content": question}]
        print("\nassistant> ", end="", flush=True)
        try:
            answer = request(candidate, args.host, args.temperature, args.reserve, args.num_ctx)
        except (ValueError, RuntimeError) as error:
            print(f"\nCould not answer: {error}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\nAnswer interrupted; it was not added to the chat.", file=sys.stderr)
            continue
        print()
        messages.extend(({"role": "user", "content": question}, {"role": "assistant", "content": answer}))
        exchanges.append((question, answer))

    if not exchanges:
        return

    destination = None
    if args.save_summary is not None:
        destination = Path(args.save_summary).expanduser().resolve() if args.save_summary else default_summary_path(paths)
    elif not args.no_save_prompt and sys.stdin.isatty() and len(exchanges) > last_saved_count:
        try:
            choice = input("Save a summary of this chat? [y/N/path] ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""
            print()
        if choice.lower() in {"y", "yes"}:
            destination = default_summary_path(paths)
        elif choice and choice.lower() not in {"n", "no"}:
            destination = Path(choice).expanduser().resolve()

    if destination:
        try:
            saved = save_summary(paths, exchanges, destination, args.host, args.temperature, args.reserve, args.num_ctx)
            print(f"Saved chat summary to {saved}")
        except (ValueError, RuntimeError) as error:
            raise SystemExit(f"Could not save chat summary: {error}") from error


if __name__ == "__main__":
    main()

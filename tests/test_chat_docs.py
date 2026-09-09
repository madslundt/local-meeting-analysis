import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "bin" / "lib" / "chat_docs.py"
SPEC = importlib.util.spec_from_file_location("chat_docs", MODULE_PATH)
chat_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_docs)


class ChatDocsTests(unittest.TestCase):
    @mock.patch.object(chat_docs, "request", return_value="Reviewed.")
    @mock.patch("builtins.input")
    @mock.patch.object(chat_docs.select, "select")
    def test_buffered_multiline_paste_is_sent_as_one_turn(self, ready, terminal_input, request):
        terminal_input.side_effect = [
            "Is this concise and neutral?",
            '\"\"\"',
            "Jan shows pragmatic judgment.",
            "Radoslaw implemented clear requirements.",
            '\"\"\"',
            "/quit",
        ]
        ready.side_effect = [
            ([object()], [], []),
            ([object()], [], []),
            ([object()], [], []),
            ([object()], [], []),
            ([], [], []),
            ([], [], []),
        ]
        with tempfile.TemporaryDirectory() as temp:
            document = Path(temp) / "summary.md"
            document.write_text("Source material")

            with mock.patch.object(chat_docs.sys, "stdin") as stdin:
                stdin.isatty.return_value = True
                chat_docs.main([str(document), "--no-save-prompt"])

        self.assertEqual(request.call_count, 1)
        sent_question = request.call_args.args[0][-1]["content"]
        self.assertEqual(
            sent_question,
            "Is this concise and neutral?\n\"\"\"\n"
            "Jan shows pragmatic judgment.\n"
            "Radoslaw implemented clear requirements.\n\"\"\"",
        )

    @mock.patch("builtins.input", return_value="one line")
    def test_noninteractive_input_remains_line_oriented(self, terminal_input):
        with mock.patch.object(chat_docs.sys, "stdin") as stdin:
            stdin.isatty.return_value = False

            question = chat_docs.read_question(input_fn=terminal_input)

        self.assertEqual(question, "one line")
        terminal_input.assert_called_once()

    def test_discovery_prefers_labelled_transcript_and_includes_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dirname in chat_docs.GENERATED_DIRS:
                (root / dirname).mkdir()
            (root / "summaries" / "meeting_summary.md").write_text("summary")
            (root / "summaries" / "ignore.txt").write_text("ignore")
            (root / "transcripts").mkdir()
            raw = root / "transcripts" / "meeting_transcript.txt"
            labelled = root / "transcripts" / "meeting_labelled_transcript.txt"
            raw.write_text("raw")
            labelled.write_text("labelled")

            found = chat_docs.discover_documents(root)

            self.assertEqual(
                found,
                [labelled, root / "summaries" / "meeting_summary.md"],
            )

    def test_document_message_labels_each_source(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "one.md"
            second = Path(temp) / "two.md"
            first.write_text("First contents")
            second.write_text("Second contents")

            message = chat_docs.document_message([first, second])

            self.assertIn("name='one.md'", message)
            self.assertIn("First contents", message)
            self.assertIn("name='two.md'", message)

    def test_all_choice_excludes_transcripts(self):
        documents = [
            Path("transcripts/meeting_labelled_transcript.txt"),
            Path("summaries/meeting_summary.md"),
            Path("evaluations/meeting_evaluation.md"),
            Path("comparisons/comparison.md"),
        ]

        selected = chat_docs.documents_for_all(documents)

        self.assertEqual(selected, documents[1:])

    def test_request_rejects_context_overflow_before_network_call(self):
        messages = [{"role": "user", "content": "x" * 500}]

        with self.assertRaisesRegex(ValueError, "NUM_CTX"):
            chat_docs.request(messages, "http://unused", 0.4, 100, 120)

    def test_chat_requests_disable_model_reasoning(self):
        body = chat_docs.completion_body([], 0.4, 2000)

        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    @mock.patch.object(chat_docs, "request", return_value="# Recap\n\nUseful finding.")
    def test_save_summary_writes_source_metadata_and_recap(self, _request):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "recap.md"
            chat_docs.save_summary(
                [Path("meeting_summary.md")],
                [("What matters?", "The constraints.")],
                destination,
                "http://unused",
                0.4,
                2000,
                32768,
            )

            text = destination.read_text()
            self.assertIn("Source documents: meeting_summary.md", text)
            self.assertIn("# Recap", text)
            self.assertFalse(destination.with_name("recap.md.part").exists())

    def test_default_summary_path_is_stable_and_safe(self):
        path = chat_docs.default_summary_path(
            [Path("My meeting (final).md"), Path("other.md")],
            datetime(2026, 9, 9, 14, 5, 6),
        )

        self.assertEqual(
            path.name,
            "2026-09-09_140506_My-meeting-final-and-1-more_chat-summary.md",
        )


if __name__ == "__main__":
    unittest.main()

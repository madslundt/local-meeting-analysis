import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "bin" / "lib" / "llm_run.py"
SPEC = importlib.util.spec_from_file_location("llm_run", MODULE_PATH)
llm_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(llm_run)


class FakeResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n".encode()
        yield b"data: [DONE]\n"


class CompletionRequestTests(unittest.TestCase):
    def test_pipeline_requests_disable_model_reasoning(self):
        body = llm_run.completion_body("system", "prompt", 0.2, 4000)

        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    @mock.patch.object(llm_run.urllib.request, "urlopen")
    def test_stream_tracks_separate_reasoning_without_saving_it(self, urlopen):
        urlopen.return_value = FakeResponse(
            [
                {"choices": [{"delta": {"reasoning_content": "scratch"}}]},
                {
                    "choices": [
                        {"delta": {"content": "answer"}, "finish_reason": "stop"}
                    ]
                },
            ]
        )

        pieces = list(llm_run.stream_completion("http://local", {}))

        self.assertEqual(pieces, ["answer"])
        self.assertTrue(llm_run.stream_completion.saw_reasoning)


if __name__ == "__main__":
    unittest.main()

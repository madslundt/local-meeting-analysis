import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "lib" / "transcribe_hf.py"
SPEC = importlib.util.spec_from_file_location("transcribe_hf", MODULE_PATH)
transcribe_hf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcribe_hf)


class BuildChunksTests(unittest.TestCase):
    def test_joins_adjacent_turns_from_same_speaker(self):
        turns = [
            {"start": 1.0, "end": 2.0, "speaker": 0},
            {"start": 2.2, "end": 4.0, "speaker": 0},
            {"start": 4.1, "end": 5.0, "speaker": 1},
        ]
        self.assertEqual(
            transcribe_hf.build_chunks(turns),
            [
                {"start": 1.0, "end": 4.0, "speaker": 0},
                {"start": 4.1, "end": 5.0, "speaker": 1},
            ],
        )

    def test_splits_long_turns(self):
        self.assertEqual(
            transcribe_hf.build_chunks([{"start": 5, "end": 70, "speaker": 0}]),
            [
                {"start": 5.0, "end": 35.0, "speaker": 0},
                {"start": 35.0, "end": 65.0, "speaker": 0},
                {"start": 65.0, "end": 70.0, "speaker": 0},
            ],
        )


class TranscribeTests(unittest.TestCase):
    def test_retries_models_that_do_not_accept_a_language_hint(self):
        calls = []

        def pipe(inputs, **kwargs):
            calls.append(kwargs)
            if "generate_kwargs" in kwargs:
                raise ValueError("unused model_kwargs: language")
            return [{"text": "hej"}]

        self.assertEqual(
            transcribe_hf.transcribe(pipe, [{}], "da", 3),
            [{"text": "hej"}],
        )
        self.assertEqual(calls[0]["generate_kwargs"], {"language": "da"})
        self.assertNotIn("generate_kwargs", calls[1])


if __name__ == "__main__":
    unittest.main()

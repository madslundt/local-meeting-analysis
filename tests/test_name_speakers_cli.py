import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class NameSpeakersCliTests(unittest.TestCase):
    def test_fresh_accepts_legacy_whisper_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "bin").mkdir()
            (project / ".venv" / "bin").mkdir(parents=True)
            shutil.copy(ROOT / "config.sh", project / "config.sh")
            shutil.copy(
                ROOT / "bin" / "name-speakers.sh",
                project / "bin" / "name-speakers.sh",
            )

            fake_python = project / ".venv" / "bin" / "python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *name_speakers.py*) exit 0 ;;\n"
                "  *merge.py*) echo '[00:00:00] Name: Hello' ;;\n"
                "esac\n"
            )
            fake_python.chmod(0o755)

            work = project / "work"
            work.mkdir()
            base = work / "recording"
            base.with_suffix(".diarization.json").write_text("{}")
            base.with_suffix(".whisper.json").write_text("{}")
            base.with_suffix(".16k.wav").write_bytes(b"")

            result = subprocess.run(
                [project / "bin" / "name-speakers.sh", "--fresh"],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("named 1, skipped 0", result.stdout)
        self.assertNotIn("no cached ASR output", result.stderr)


if __name__ == "__main__":
    unittest.main()

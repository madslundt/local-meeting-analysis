import importlib.util
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "bin" / "lib" / "name_speakers.py"
SPEC = importlib.util.spec_from_file_location("name_speakers", MODULE_PATH)
name_speakers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(name_speakers)


class FakePlayer:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def kill(self):
        raise AssertionError("responsive player should not need to be killed")

    def wait(self, timeout=None):
        return -15 if self.terminated else 0


class NameSpeakersTests(unittest.TestCase):
    def test_keypress_stops_player_without_consuming_terminal_input(self):
        player = FakePlayer()
        terminal = mock.Mock()
        terminal.isatty.return_value = True
        terminal.fileno.return_value = 12
        original = [0, 0, 0, name_speakers.termios.ICANON |
                    name_speakers.termios.ECHO]

        with (
            mock.patch.object(name_speakers.subprocess, "Popen",
                              return_value=player),
            mock.patch.object(name_speakers.sys, "stdin", terminal),
            mock.patch.object(name_speakers.termios, "tcgetattr",
                              side_effect=[original.copy(), original.copy()]),
            mock.patch.object(name_speakers.termios, "tcsetattr") as set_attrs,
            mock.patch.object(name_speakers.select, "select",
                              return_value=([terminal], [], [])),
        ):
            interrupted = name_speakers.play_until_keypress(
                ["afplay", "clip.wav"]
            )

        self.assertTrue(interrupted)
        self.assertTrue(player.terminated)
        terminal.read.assert_not_called()
        immediate = set_attrs.call_args_list[0].args[2]
        self.assertFalse(immediate[3] & name_speakers.termios.ICANON)
        self.assertFalse(immediate[3] & name_speakers.termios.ECHO)
        self.assertEqual(set_attrs.call_args_list[-1].args[2], original)

if __name__ == "__main__":
    unittest.main()

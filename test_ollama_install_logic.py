
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Mock necessary modules for import
sys.modules['gradio'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Avoid relative import in ollama.py
# Temporary hack: mock prompt_guides if it's imported relatively
# Or better, just fix the script to run as a package.

# Let's try mocking the relative import by setting up a dummy package
import types
dummy_pkg = types.ModuleType('smooth_brain')
dummy_pkg.__path__ = []
sys.modules['smooth_brain'] = dummy_pkg
sys.modules['smooth_brain.prompt_guides'] = MagicMock()

import ollama

class TestOllamaInstallLogic(unittest.TestCase):

    def test_find_ollama_local_priority(self):
        """Test that _find_ollama prioritizes local bin/ directory."""
        with patch('os.path.exists') as mock_exists, \
             patch('shutil.which') as mock_which:

            # Scenario: Both local and system exist
            # mock_exists returns True for local, mock_which returns system path
            mock_exists.side_effect = lambda p: "bin" in p
            mock_which.return_value = "/usr/bin/ollama"

            path = ollama._find_ollama()
            # On Windows, _find_ollama will add .exe, so we use assertIn or just check the tail
            self.assertTrue(path.endswith("ollama") or path.endswith("ollama.exe"))
            self.assertIn("bin", path)

    def test_get_official_checksums(self):
        """Test fetching official checksums from GitHub."""
        checksums = ollama._get_official_checksums()
        self.assertIsInstance(checksums, dict)
        if checksums:
            # If we got data, check for common platforms
            self.assertTrue(any("linux-amd64" in k for k in checksums) or
                            any("windows-amd64" in k for k in checksums))
            print(f"\nSuccessfully fetched {len(checksums)} checksums from GitHub.")
        else:
            print("\nWarning: Could not fetch checksums (possibly rate limited or no internet).")

    def test_sanitize_prompt(self):
        """Test prompt sanitization logic."""
        input_prompt = "```\nHello\n\n\nWorld\n```"
        # The actual function uses strip() and replace('\n\n\n', '\n\n')
        # sanitize_prompt(p) -> p.replace('```', '').replace('\n\n\n', '\n\n').strip()
        expected = "Hello\n\nWorld"
        self.assertEqual(ollama.sanitize_prompt(input_prompt), expected)

if __name__ == '__main__':
    unittest.main()

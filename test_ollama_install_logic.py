import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Setup mocks for relative imports before importing ollama
sys.modules['smooth_brain.prompt_guides'] = MagicMock()
sys.modules['smooth_brain.story_templates'] = MagicMock()

# Mock other heavy deps
sys.modules['gradio'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Re-mock the plugin package structure if needed
class MockPackage:
    __path__ = []
sys.modules['.'] = MockPackage()

import ollama

class TestOllamaRobust(unittest.TestCase):

    def test_sanitize_prompt(self):
        self.assertEqual(ollama.sanitize_prompt("```json\nhello\n```"), "hello")
        self.assertEqual(ollama.sanitize_prompt("a\n\n\n\nb"), "a\n\nb")

    @patch('ollama._BIN_DIR', '/tmp/sb_bin')
    @patch('os.path.exists')
    @patch('shutil.which')
    def test_find_ollama_priority(self, mock_which, mock_exists):
        # Local exists
        mock_exists.side_effect = lambda p: p == os.path.join('/tmp/sb_bin', 'ollama')
        self.assertEqual(ollama._find_ollama(), os.path.join('/tmp/sb_bin', 'ollama'))

        # Local missing, system exists
        mock_exists.side_effect = lambda p: False
        mock_which.return_value = '/usr/bin/ollama'
        self.assertEqual(ollama._find_ollama(), '/usr/bin/ollama')

    @patch('urllib.request.urlopen')
    def test_get_official_checksums_mocked(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"hash1 filename1\nhash2 filename2"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        sums = ollama._get_official_checksums()
        self.assertEqual(sums, {"filename1": "hash1", "filename2": "hash2"})

    def test_verify_sha256(self):
        # Test with a real temporary file
        test_file = "/tmp/sb_test_verify"
        content = b"hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        with open(test_file, "wb") as f:
            f.write(content)

        try:
            self.assertTrue(ollama._verify_sha256(test_file, expected))
            self.assertFalse(ollama._verify_sha256(test_file, "wrong"))
        finally:
            if os.path.exists(test_file): os.remove(test_file)

if __name__ == '__main__':
    unittest.main()

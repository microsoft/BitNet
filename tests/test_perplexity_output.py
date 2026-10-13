import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("perplexity", Path(__file__).resolve().parents[1] / "utils/test_perplexity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PerplexityOutputTests(unittest.TestCase):
    def setUp(self):
        self.tester = module.PerplexityTester.__new__(module.PerplexityTester)

    def test_final_estimate_wins_over_progress(self):
        self.assertEqual(self.tester.parse_perplexity("PPL = 20.0 +/- 2.0\nFinal estimate: PPL = 8.25 +/- 0.5"), "8.2500±0.5000")

    def test_last_result_without_final_marker(self):
        self.assertEqual(self.tester.parse_perplexity("PPL = 20\nPPL = 9"), "9.0000")

    def test_scientific_notation_and_unicode_uncertainty(self):
        self.assertEqual(self.tester.parse_perplexity("Final estimate: PPL = 1.25e2 ± 2.5e-1"), "125.0000±0.2500")

    def test_missing_estimate(self):
        self.assertIsNone(self.tester.parse_perplexity("model loaded"))

if __name__ == "__main__":
    unittest.main()

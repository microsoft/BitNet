import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).parents[1]))
from pack_weight import convert_weight_int8_to_int2

class PackingTests(unittest.TestCase):
    def test_non_ternary_values_are_not_silently_encoded(self):
        for value in [-2, 2, 127]:
            with self.assertRaisesRegex(ValueError, "ternary"):
                convert_weight_int8_to_int2(torch.full((16,32), value, dtype=torch.int8))

    def test_floating_weights_require_explicit_quantization(self):
        with self.assertRaisesRegex(ValueError, "int8"):
            convert_weight_int8_to_int2(torch.zeros(16,32))

    def test_tile_boundaries_are_validated(self):
        for shape in [(15,32), (16,31), (16,), (16,32,1)]:
            with self.assertRaisesRegex(ValueError, "shape"):
                convert_weight_int8_to_int2(torch.zeros(shape, dtype=torch.int8))

    def test_valid_codes_are_preserved(self):
        for value, byte in [(-1,85), (0,-86), (1,-1)]:
            result = convert_weight_int8_to_int2(torch.full((16,32), value, dtype=torch.int8))
            self.assertEqual(result.shape, (16,8))
            self.assertTrue(torch.all(result == byte))

if __name__ == "__main__": unittest.main()

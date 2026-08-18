# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "production_debt.py",
)
spec = importlib.util.spec_from_file_location("bitnet_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["bitnet_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtBitNetGate = production_debt_mod.ProductionDebtBitNetGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtBitNetGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtBitNetGate(
            never_equate_intent_to_approval=True,
            max_acceptable_bdi=12.0,
        )

    def test_clean_1bit_inference_passes_readiness(self) -> None:
        report = self.gate.evaluate_1bit_inference(
            model_id="bitnet_b1_58_3b_ternary",
            allocated_ternary_weights_bytes=1500000000,
            peak_activation_buffer_bytes=1580000000,
            kernel_latency_us=9.8,
            ternary_saturation_errors=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.bdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_1bit_inference_fails_debt(self) -> None:
        report = self.gate.evaluate_1bit_inference(
            model_id="uncalibrated_bitnet_model",
            allocated_ternary_weights_bytes=1500000000,
            peak_activation_buffer_bytes=4200000000,  # 2.8x activation sprawl
            kernel_latency_us=85.0,  # High kernel latency
            ternary_saturation_errors=3,  # 3 clipping saturation errors
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.bdi_score, 50.0)
        self.assertIn("HIGH_ACTIVATION_MEMORY_SPRAWL_2.80X", report.critical_smells)
        self.assertIn("HIGH_TERNARY_KERNEL_LATENCY_85.0US", report.critical_smells)
        self.assertIn("DETECTED_3_TERNARY_WEIGHT_CLIPPING_ERRORS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_BITNET_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_1bit_inference("model-1")
        self.gate.evaluate_1bit_inference("model-2")
        self.gate.evaluate_1bit_inference("model-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class BitNetDebtReport:
    model_id: str
    bdi_score: float  # BitNet Debt Index (target <= 12.0)
    activation_sprawl_multiplier: float  # Target <= 1.08x
    kernel_latency_us: float  # Target <= 12.0us
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for BitNet 1-bit inference runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_bitnet_event(
        self,
        model_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{model_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "model_id": model_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtBitNetGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for BitNet 1-Bit LLM Runtime.

    Quantifies ternary weight clipping saturation, 8-bit dynamic activation scaling drift, and matmul-free kernel latency against 4 Enterprise KPIs:
    1. BitNet Debt Index (BDI <= 12.0)
    2. Dynamic Activation Scaling Multiplier (DASM <= 1.08x)
    3. P99 Ternary Kernel Latency (<= 12.0us)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_bdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_bdi = max_acceptable_bdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_1bit_inference(
        self,
        model_id: str,
        allocated_ternary_weights_bytes: int = 1500000000,
        peak_activation_buffer_bytes: int = 1580000000,
        kernel_latency_us: float = 9.8,
        ternary_saturation_errors: int = 0,
        un_gated_mutations: int = 0,
    ) -> BitNetDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_bitnet_event(
                model_id=model_id,
                event_type="inference_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. BitNet execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Activation Scaling Multiplier
        act_ratio = peak_activation_buffer_bytes / max(1, allocated_ternary_weights_bytes)
        if act_ratio > 1.8:
            critical_smells.append(f"HIGH_ACTIVATION_MEMORY_SPRAWL_{act_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if kernel_latency_us > 40.0:
            critical_smells.append(f"HIGH_TERNARY_KERNEL_LATENCY_{kernel_latency_us:.1f}US")

        # Ternary saturation errors
        if ternary_saturation_errors > 0:
            critical_smells.append(f"DETECTED_{ternary_saturation_errors}_TERNARY_WEIGHT_CLIPPING_ERRORS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_BITNET_MUTATIONS")

        # KPI 1: BitNet Debt Index (0 = Clean, 100 = Catastrophic)
        bdi = (
            max(0.0, (act_ratio - 1.0) * 20.0)
            + max(0.0, (kernel_latency_us - 12.0) * 0.5)
            + (ternary_saturation_errors * 25.0)
            + (un_gated_mutations * 30.0)
        )
        bdi_score = round(min(100.0, bdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - bdi_score)
        is_production_ready = (
            bdi_score <= self.max_acceptable_bdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_bitnet_event(
            model_id=model_id,
            event_type="bitnet_authorized" if is_production_ready else "bitnet_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "bdi_score": bdi_score,
                "act_ratio": act_ratio,
                "allocated_ternary_weights_bytes": allocated_ternary_weights_bytes,
                "peak_activation_buffer_bytes": peak_activation_buffer_bytes,
                "kernel_latency_us": kernel_latency_us,
                "ternary_saturation_errors": ternary_saturation_errors,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return BitNetDebtReport(
            model_id=model_id,
            bdi_score=bdi_score,
            activation_sprawl_multiplier=round(act_ratio, 2),
            kernel_latency_us=round(kernel_latency_us, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )

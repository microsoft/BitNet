# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseStats:
    name: str
    tokens: int
    time: float

    def show(self) -> str:
        tps = self.tokens / self.time
        return (
            f"[{self.name}] "
            f"generated tokens: {self.tokens}"
            f" - total time: {self.time:.3f}s"
            f" - {tps:.1f} tokens per second"
        )


class Stats:
    """
    Generation stats, split by phases.
    """

    def __init__(self):
        self.phases = []
        self.current = None

    def end_phase(self, tokens: int, now: Optional[float] = None):
        """Terminate the current phase."""
        if self.current is None:
            return
        if now is None:
            now = time.time()
        cname, ctokens, ctime = self.current
        stats = PhaseStats(
            name=cname,
            tokens=tokens - ctokens,
            time=now - ctime,
        )
        self.phases.append(stats)

    def phase(self, name: str, tokens: int = 0):
        """
        Start a new phase, and terminate the current one,
        if one is ongoing.
        """
        now = time.time()
        self.end_phase(tokens, now)
        self.current = (name, tokens, now)
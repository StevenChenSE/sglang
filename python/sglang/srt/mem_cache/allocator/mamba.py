"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Iterator, Optional

import torch


def _guard_violation(site: str, detail: str):
    """Persist + raise on a slot-accounting violation (double free, range
    breach, duplicate handout). The write goes to $HOME so it survives the
    crash handler; raising here surfaces the full host stack in the server
    log while the GPU is still healthy."""
    stamp = time.strftime("%H:%M:%S")
    try:
        with open(os.path.expanduser("~/mamba_alloc_guard.log"), "a") as f:
            f.write(f"\n==== [{stamp}] MAMBA ALLOC GUARD [{site}]\n{detail}\n")
            f.write("".join(traceback.format_stack()[-25:]))
    except Exception:
        pass
    raise RuntimeError(f"Mamba alloc guard [{site}]: {detail}")


class MambaSlotAllocator:
    """Free-list of Mamba pool slot indices. Deliberately not a subclass of
    ``BaseTokenToKVPoolAllocator``: slots are per request, not per token."""

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # Set by alloc_group_begin(); alloc(1) drains it until alloc_group_end().
        self._alloc_iter: Optional[Iterator] = None
        # SGL_MAMBA_ALLOC_GUARD=1: track occupancy so free()/alloc() catch
        # double frees, out-of-range ids, and duplicate handouts at the exact
        # call site (the free() path is a blind torch.cat otherwise).
        from sglang.srt.utils.flight_flags import flag_on

        self._guard = (
            flag_on("MAMBA_ALLOC_GUARD")
            or os.environ.get("SGL_MAMBA_ALLOC_GUARD", "0") == "1"
        )
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Same as ``available_size`` for a static pool;
        byte-coordinated allocators return their byte-limited view instead."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        if self._guard:
            ids = select_index.to(device=self.device, dtype=torch.long)
            if ((ids < 1) | (ids > self.size)).any().item():
                _guard_violation(
                    "alloc-range", f"free list handed out out-of-range ids {ids.tolist()}"
                )
            dup = self._occupied[ids]
            if dup.any().item():
                _guard_violation(
                    "alloc-double",
                    f"free list handed out in-use slots {ids[dup].tolist()} "
                    f"(free-list over-count / duplicate id)",
                )
            self._occupied[ids] = True
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self._guard:
            ids = free_index.to(device=self.device, dtype=torch.long)
            if ((ids < 0) | (ids > self.size)).any().item():
                _guard_violation(
                    "free-range",
                    f"freeing out-of-range slot ids {ids.tolist()} (a -1 or "
                    f"garbage id entered the free list)",
                )
            dead = ~self._occupied[ids]
            if dead.any().item():
                _guard_violation(
                    "free-double",
                    f"freeing non-occupied slots {ids[dead].tolist()} "
                    f"(double free / free-while-referenced)",
                )
            self._occupied[ids] = False
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        if self._guard:
            self._occupied = torch.zeros(
                self.size + 1, dtype=torch.bool, device=self.device
            )

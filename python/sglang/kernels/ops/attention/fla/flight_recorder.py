"""12.152 Flight recorder for the deep-context GPU fault.

Passive, host-side launch tracing into a pinned-memory ring that survives GPU
context death. For each instrumented kernel launch the HOST writes a header
(kernel id, sequence, tensor data_ptrs, key extents) BEFORE the launch — no
device atomics, no kernel modifications. A daemon thread snapshots the
allocator's segment map and persists everything to $HOME every 2 s, so a
fault (clean SIGABRT death or hard freeze) leaves a receipt naming the last
launches and their pointer arguments.

Post-fault analysis (scripts/analyze_flight.py) matches each recorded
data_ptr against the last allocator snapshot: a pointer outside every live
segment is the dangling buffer, and the kernel id + arg position identify the
exact tensor that died.

Env: SGL_FLIGHT_RECORDER=1 (default off — zero overhead).
"""

from __future__ import annotations

import os
import threading
import time

import torch

MAGIC = 0x464C5231  # "FLR1"
SLOTS = 4096
REC_U32 = 32  # 128 B per record

# 12.165: pinned WMMA flight planes, installed by gptq_kernels via
# set_wmma_planes(); dumped next to the ring by the persist thread.
_WMMA_PLANES = None


def set_wmma_planes(t) -> None:
    global _WMMA_PLANES
    _WMMA_PLANES = t

# record layout (u32 offsets within a record)
OFF_MAGIC = 0
OFF_KID = 1
OFF_SEQ_LO = 2
OFF_SEQ_HI = 3
OFF_PTRS = 4  # 8 tensors x u64 = 16 u32  [4..19]
OFF_EXT = 20  # 8 extents [20..27]
OFF_TS_LO = 28  # host timestamp (s) u64 [28..29]
OFF_DONE = 29  # 1 = the host observed the NEXT launch (execution likely done)
OFF_VER = 30
# [24..31] reserved (future in-kernel payload / done-stamp)

REC_BYTES = REC_U32 * 4


class FlightRecorder:
    def __init__(self):
        from sglang.srt.utils.flight_flags import flag_on

        self.enabled = (
            flag_on("FLIGHT_RECORDER")
            or os.getenv("SGL_FLIGHT_RECORDER", "0") == "1"
        )
        if not self.enabled:
            return
        self.ring = torch.zeros(SLOTS * REC_U32, dtype=torch.int32, pin_memory=True)
        self.names: dict[int, str] = {}
        self.seq = 0
        self.last_snapshot = None
        self.snapshot_seq = -1
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._persist_loop, daemon=True)
        self._thread.start()

    # ---- launch-time API (host-side only, no sync, no GPU work) ----------
    def record(self, kernel_id: int, tensors=(), extents=(), name: str = "") -> int:
        self.names.setdefault(kernel_id, name or f"kernel{kernel_id}")
        with self._lock:
            # mark the PREVIOUS launch done before allocating this slot
            if self.seq > 0:
                self.ring[(self.seq - 1) % SLOTS * REC_U32 + OFF_DONE] = 1
            seq = self.seq
            self.seq += 1
            slot = seq % SLOTS
            base = slot * REC_U32
            r = self.ring
            r[base + OFF_MAGIC] = MAGIC
            r[base + OFF_KID] = kernel_id
            r[base + OFF_SEQ_LO] = seq & 0xFFFFFFFF
            r[base + OFF_SEQ_HI] = (seq >> 32) & 0xFFFFFFFF
            # clear stale payload from the previous record in this slot
            for j in range(16):
                r[base + OFF_PTRS + j] = 0
            for j in range(8):
                r[base + OFF_EXT + j] = 0
            for j, t in enumerate(tensors[:8]):
                if t is None:
                    continue
                p = t.data_ptr()
                lo = p & 0xFFFFFFFF
                hi = (p >> 32) & 0xFFFFFFFF
                # int32 tensor: map u32 halves into signed range
                r[base + OFF_PTRS + 2 * j] = lo - (1 << 32) if lo >= (1 << 31) else lo
                r[base + OFF_PTRS + 2 * j + 1] = (
                    hi - (1 << 32) if hi >= (1 << 31) else hi
                )
            for j, e in enumerate(extents[:8]):
                v = int(e)
                r[base + OFF_EXT + j] = max(-2**31, min(2**31 - 1, v))
            now = time.time()
            r[base + OFF_TS_LO] = int(now) & 0xFFFFFFFF
            r[base + OFF_DONE] = 0
            r[base + 24] = 0  # entered marker (sticky — clear per record)
            r[base + 25] = 0  # exited marker (sticky — clear per record)
            return slot

    def mark_prev_done(self) -> None:
        """Called at the NEXT launch: the previous launch has at least started
        (same-stream ordering); under a single queue it has usually finished."""
        with self._lock:
            if self.seq == 0:
                return
            prev = (self.seq - 1) % SLOTS
            self.ring[prev * REC_U32 + OFF_DONE] = 1

    # ---- persistence -----------------------------------------------------
    def _snapshot_allocator(self):
        try:
            snap = torch.cuda.memory_snapshot()
            self.last_snapshot = [
                {
                    "device": s.get("device", 0),
                    "address": s.get("address", 0),
                    "size": s.get("total_size", s.get("size", 0)),
                }
                for s in snap
            ]
            self.snapshot_seq = self.seq
        except Exception:
            pass

    def _persist_loop(self):
        out = os.path.expanduser("~/flight_recorder_latest.bin")
        meta = os.path.expanduser("~/flight_recorder_meta.txt")
        planes_out = os.path.expanduser("~/wmma_flight_planes.bin")
        # 12.168: per-process heartbeat — append-only with ts+seq so a freeze
        # leaves an exact host-alive timeline per rank (meta is shared/last-win).
        hb = os.path.expanduser(f"~/hb_pid{os.getpid()}.txt")
        # startup marker: proves the thread is alive even before any record()
        try:
            with open(meta, "w") as f:
                f.write(f"persist-thread-started {time.time()}\n")
        except Exception:
            pass
        while not self._stop:
            # 0) per-process heartbeat (12.168)
            try:
                if not os.path.exists(hb) or os.path.getsize(hb) > (1 << 21):
                    mode = "w"
                else:
                    mode = "a"
                with open(hb, mode) as f:
                    f.write(f"hb {time.time():.3f} seq={self.seq}\n")
            except Exception:
                pass
            # 1) THE RING IS THE CRITICAL PAYLOAD — write it in its own try,
            #    never gated on the allocator snapshot succeeding.
            try:
                with self._lock:
                    seq = self.seq
                    blob = self.ring.numpy().tobytes()
                    names = dict(self.names)
                with open(out, "wb") as f:
                    f.write(blob)
                with open(meta, "w") as f:
                    f.write(f"seq={seq}\n")
                    f.write(f"snapshot_seq={self.snapshot_seq}\n")
                    for kid, nm in names.items():
                        f.write(f"kernel {kid} = {nm}\n")
            except Exception:
                pass
            # 1b) WMMA flight planes (12.165): pinned host, readable even
            #     while the device is wedged; last snapshot = wedge state.
            if _WMMA_PLANES is not None:
                try:
                    with open(planes_out, "wb") as f:
                        f.write(_WMMA_PLANES.numpy().tobytes())
                except Exception:
                    pass
            # 2) host mmap map: settles faulting-VA class (host vs device)
            try:
                with open("/proc/self/maps") as f:
                    lines = f.readlines()
                with open(out + ".hostmaps", "w") as f:
                    f.writelines(lines[-4000:])
            except Exception:
                pass
            # 3) allocator snapshot: best-effort, separate failure domain
            try:
                self._snapshot_allocator()
                snap = self.last_snapshot
                if snap:
                    with open(out + ".segments", "w") as f:
                        for s in snap:
                            f.write(f"{s['device']} {s['address']} {s['size']}\n")
            except Exception:
                pass
            time.sleep(2.0)


_REC: FlightRecorder | None = None
_REC_LOCK = threading.Lock()


def get_recorder() -> FlightRecorder:
    global _REC
    with _REC_LOCK:
        if _REC is None:
            _REC = FlightRecorder()
        return _REC


def record(kernel_id: int, tensors=(), extents=(), name: str = "") -> None:
    """Module-level convenience: no-op unless the flight recorder is on."""
    record_with_ring(kernel_id, tensors, extents, name)


def record_with_ring(kernel_id: int, tensors=(), extents=(), name: str = ""):
    """Record a launch; return (pinned ring tensor, slot) for in-kernel
    entered/exited markers. Returns (None, None) when disabled."""
    rec = get_recorder()
    if not rec.enabled:
        return None, None
    slot = rec.record(kernel_id, tensors, extents, name)
    return rec.ring, slot

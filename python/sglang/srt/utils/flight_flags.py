"""File-based instrument flags (12.156b).

The spawn-child environment proved unreliable for ad-hoc instrument flags
(exportable_env_vars filters to registered Envs fields), so the boot script
writes ~/flight_recorder_flags.txt and every instrument reads THIS. Zero GPU
cost; read once per process and cached.
"""

from __future__ import annotations

import os
from typing import Optional

_FLAG_PATH = os.path.expanduser("~/flight_recorder_flags.txt")
_cache: Optional[dict] = None
_lock = None


def _load() -> dict:
    global _cache
    if _cache is None:
        flags = {}
        try:
            with open(_FLAG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    flags[k.strip()] = v.strip()
        except Exception:
            pass
        _cache = flags
    return _cache


def get_flag(name: str, default: str = "") -> str:
    """Instrument flag: flags file first, then the environment, then default."""
    v = _load().get(name)
    if v is not None:
        return v
    return os.environ.get(name, default)


def flag_on(name: str) -> bool:
    return get_flag(name, "0") == "1"


_bisect_n = 0


def bisect_sync(tag: str) -> None:
    """SGLANG_SYNC_BISECT=1: sync after each suspect enqueue (§12.164).

    The silent wedge raises no interrupt, so the kernel log cannot name it;
    a sync after every launch in the suspect window does: the last
    "[BISECT:n] ... OK" line before silence is the wedge point — the first
    sync that hangs or raises localizes the wedge to the op right before it.
    Diagnostic runs only — every sync drains the pipeline.
    """
    global _bisect_n
    if not flag_on("SGLANG_SYNC_BISECT"):
        return
    import torch

    # cuda-graph capture runs the same forward code at boot; synchronize()
    # during capture is illegal and invalidates the graph
    if torch.cuda.is_current_stream_capturing():
        return
    _bisect_n += 1
    print(f"[BISECT:{_bisect_n}] sync after {tag}", flush=True)
    torch.cuda.synchronize()
    print(f"[BISECT:{_bisect_n}] sync after {tag} OK", flush=True)

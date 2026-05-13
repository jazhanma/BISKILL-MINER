"""BiSkill Miner — Bimanual Skill Retrieval Agent.

Importing this package proactively constrains BLAS/OpenMP thread counts
*before* PyTorch, FAISS, or NumPy initialize their own thread pools.
This avoids a known segfault on some Python builds where PyTorch and
FAISS race for OpenMP threads when run in the same process.
"""
from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = []

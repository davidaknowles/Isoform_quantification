"""
native.py
=========
ctypes bindings for the compiled E-step kernels in rust/joli_kernels
(build with `cargo build --release` in that directory). The library is
looked up at $JOLI_NATIVE_LIB, then at the repository's build path.
available() is False when it cannot be loaded, and callers fall back to numpy.
"""

import ctypes
import os
from pathlib import Path

import numpy as np

_LIB = None
_TRIED = False
_DEFAULT = Path(__file__).resolve().parent.parent / "rust" / "joli_kernels" / "target" / "release" / "libjoli_kernels.so"


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = Path(os.environ.get("JOLI_NATIVE_LIB", _DEFAULT))
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    lib.joli_estep.restype = ctypes.c_double
    lib.joli_estep.argtypes = [size, ptr, ptr, ptr, ptr, ptr, size, ptr, size]
    lib.joli_estep_gather.restype = ctypes.c_double
    lib.joli_estep_gather.argtypes = [size, ptr, ptr, ptr, ptr, ptr, size, ptr, ptr, ptr, ptr, ptr, size]
    lib.joli_position_counts.restype = None
    lib.joli_position_counts.argtypes = [size, ptr, ptr, ptr, ptr, ptr, size, ptr, size]
    _LIB = lib
    return lib


def available():
    return _load() is not None


def default_threads():
    return int(os.environ.get("JOLI_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _check(ec_ptr, tx, c, counts, mu):
    for a, dt in ((ec_ptr, np.int64), (tx, np.int64), (c, np.float64), (counts, np.float64), (mu, np.float64)):
        if a.dtype != dt or not a.flags.c_contiguous:
            raise TypeError("native kernels need contiguous int64 ec_ptr/tx and float64 c/counts/mu")


def gather_index(tx, ec, n_tx):
    """Transcript-sorted index of positions for joli_estep_gather: (tx_ptr, perm, pos_ec_sorted)."""
    perm = np.argsort(tx, kind="stable").astype(np.int64)
    tx_ptr = np.r_[0, np.cumsum(np.bincount(tx, minlength=n_tx))].astype(np.int64)
    return tx_ptr, perm, np.ascontiguousarray(ec[perm], dtype=np.int64)


def estep(ec_ptr, tx, c, counts, mu, threads=None, index=None):
    """
    Returns (g, ll): g (T,) with g_t = sum_e n_e c_et / denom_e and
    ll = sum_e n_e log denom_e, over ECs with denom_e > 0. With more than one
    thread and a gather_index, runs the accumulator-free two-pass kernel.
    """
    lib = _load()
    _check(ec_ptr, tx, c, counts, mu)
    threads = threads or default_threads()
    g = np.empty(len(mu), dtype=np.float64)
    if threads > 1 and index is not None:
        tx_ptr, perm, pos_ec = index
        ratio = np.empty(len(counts), dtype=np.float64)
        ll = lib.joli_estep_gather(len(counts), _p(ec_ptr), _p(tx), _p(c), _p(counts), _p(mu), len(mu),
                                   _p(tx_ptr), _p(perm), _p(pos_ec), _p(ratio), _p(g), threads)
    else:
        ll = lib.joli_estep(len(counts), _p(ec_ptr), _p(tx), _p(c), _p(counts), _p(mu), len(mu), _p(g), threads)
    return g, ll


def position_counts(ec_ptr, tx, c, counts, mu, threads=None):
    """z (n_pos,) with z_p = n_e mu_t c_p / denom_e (0 where denom_e = 0)."""
    lib = _load()
    _check(ec_ptr, tx, c, counts, mu)
    z = np.empty(len(tx), dtype=np.float64)
    lib.joli_position_counts(len(counts), _p(ec_ptr), _p(tx), _p(c), _p(counts), _p(mu), len(mu), _p(z),
                             threads or default_threads())
    return z

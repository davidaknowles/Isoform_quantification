"""
rad.py
======
Read-position data from single-cell RAD files, as dumped by `rust/radpos`
(flat little-endian columns per read and per alignment), for positional bias
models of short-read single-cell libraries such as Parse.

A molecule is a (cell barcode, UMI) pair; one read represents it (the first
in the file). Its alignments are filtered to best-score ties and collapsed
into equivalence classes keyed by transcript set and binned read position,
reusing paf.library_from_alignments, so that a model of the read-position
distribution can reweight compatibilities exactly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paf import library_from_alignments

BASES = np.array(list("ACGT"))


def load_rad_dump(path):
    """
    Memory-map a radpos dump.

    Returns dict with meta (dict), refs (list[str]) and arrays:
      read_bc, read_umi (n_reads,) uint64; read_naln (n_reads,) uint32;
      aln_ref, aln_start, aln_end, aln_tlen (n_alns,) uint32; aln_fw (n_alns,) uint8;
      aln_score (n_alns,) int32; aln_read (n_alns,) int64 read index of each alignment.
    """
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    out = {"meta": meta, "refs": (path / "refs.txt").read_text().split("\n")[: meta["n_refs"]]}
    files = {"read_bc": ("read_bc.u64", "<u8"), "read_umi": ("read_umi.u64", "<u8"),
             "read_naln": ("read_naln.u32", "<u4"), "aln_ref": ("aln_ref.u32", "<u4"),
             "aln_fw": ("aln_fw.u8", "u1"), "aln_score": ("aln_score.i32", "<i4"),
             "aln_start": ("aln_start.u32", "<u4"), "aln_end": ("aln_end.u32", "<u4"),
             "aln_tlen": ("aln_tlen.u32", "<u4")}
    for name, (fname, dt) in files.items():
        n = meta["n_reads"] if name.startswith("read_") else meta["n_alns"]
        out[name] = np.memmap(path / fname, dtype=dt, mode="r", shape=(n,)) if n else np.zeros(0, dtype=dt)
    out["aln_read"] = np.repeat(np.arange(meta["n_reads"], dtype=np.int64), out["read_naln"])
    return out


def decode_2bit(codes, length):
    """2-bit packed sequences (first base in the highest bits) to strings."""
    codes = np.asarray(codes, dtype=np.uint64)
    shifts = np.arange(length - 1, -1, -1, dtype=np.uint64) * np.uint64(2)
    idx = ((codes[:, None] >> shifts[None, :]) & np.uint64(3)).astype(np.int64)
    return ["".join(r) for r in BASES[idx]]


def molecule_alignments(dump, barcode_to_cell, score_frac=1.0, fraction=1.0, seed=0):
    """
    One read per molecule and its best-score alignments.

    Args:
        dump            : load_rad_dump output.
        barcode_to_cell : Series barcode string -> integer cell index (cells not in it are dropped).
        score_frac      : keep alignments with score >= score_frac * best score of the read.
        fraction        : keep a random fraction of molecules (e.g. to fit bias profiles).
    Returns DataFrame, one row per kept alignment: read (molecule id), cell, tx_idx
    (index into dump["refs"]), start, end, tlen.
    """
    cblen = dump["meta"]["cblen"]
    ucode, inv = np.unique(np.asarray(dump["read_bc"]), return_inverse=True)
    cell_of_code = barcode_to_cell.reindex(decode_2bit(ucode, cblen)).to_numpy(dtype=np.float64)
    read_cell = cell_of_code[inv]
    ok = ~np.isnan(read_cell)
    # first read of each (cell, UMI)
    key = pd.DataFrame({"cell": read_cell[ok].astype(np.int64), "umi": np.asarray(dump["read_umi"])[ok]})
    reads = np.flatnonzero(ok)[~key.duplicated(["cell", "umi"]).to_numpy()]
    if fraction < 1.0:
        reads = reads[np.random.default_rng(seed).random(len(reads)) < fraction]
    mol_of_read = np.full(len(read_cell), -1, dtype=np.int64)
    mol_of_read[reads] = np.arange(len(reads))
    mol = mol_of_read[dump["aln_read"]]
    sel = np.flatnonzero(mol >= 0)
    mol = mol[sel]
    score = np.asarray(dump["aln_score"])[sel]
    best = np.full(len(reads), np.iinfo(np.int32).min, dtype=np.int64)
    np.maximum.at(best, mol, score)
    keep = score >= score_frac * best[mol]
    idx, mol = sel[keep], mol[keep]
    return pd.DataFrame({
        "read": mol, "cell": read_cell[reads][mol].astype(np.int32),
        "tx_idx": np.asarray(dump["aln_ref"])[idx].astype(np.int32),
        "start": np.asarray(dump["aln_start"])[idx], "end": np.asarray(dump["aln_end"])[idx],
        "tlen": np.asarray(dump["aln_tlen"])[idx],
    })


def libraries_from_molecules(aln, cell_group, transcript_names, rates, model=None, platform="short"):
    """
    Pseudobulk libraries with per-molecule alignments, optionally keyed by a
    positional model's integer features.

    Args:
        aln              : molecule_alignments output.
        cell_group       : array (n_cells,) group label of each cell index.
        transcript_names : list[str] (T,) -- the RAD file's reference order (dump["refs"]),
                           which tx_idx indexes.
        rates            : dict {group: b (T,)} sampling rates per group.
        model            : object with features(start, end, tlen) -> (n, F) int, or None.
    Returns dict {group: Library}.
    """
    groups = np.asarray(cell_group)[aln["cell"].to_numpy()]
    out = {}
    for g in pd.unique(groups):
        a = aln[groups == g]
        feat = None if model is None else model.features(a["start"], a["end"], a["tlen"])
        out[g] = library_from_alignments(str(g), a, transcript_names, features=feat,
                                         platform=platform, b=rates[g])
    return out


_CODE = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i


def encode_2bit(seqs, length):
    """Strings of ACGT (fixed length <= 32) to 2-bit packed uint64, first base highest; -1 if any other base."""
    a = np.asarray(seqs, dtype=f"S{length}").view(np.uint8).reshape(-1, length)
    c = _CODE[a]
    bad = (c == 255).any(axis=1)
    out = np.zeros(len(a), dtype=np.uint64)
    for j in range(length):
        out = (out << np.uint64(2)) | (c[:, j] & 3).astype(np.uint64)
    out[bad] = np.iinfo(np.uint64).max
    return out

"""
G/C content of target sequences for the short-read fragment GC model (FragmentModel(gc=...)).

GCIndex stores each target's G/C indicator as packed bits (1 bit per base) and gives
  fragment_gc(tx, start, flen): G/C fraction of fragments [start, start + flen) on targets tx;
  position_hist(lengths, edges, max_pos): per target and fragment length, the share of start positions
      whose fragment falls in each G/C bin, from at most max_pos evenly spaced starts (T, len(lengths), K).
"""

import gzip

import numpy as np

_GC = np.zeros(256, dtype=bool)
_GC[[ord(c) for c in "GCgcSs"]] = True


def _read_fasta(path):
    op = gzip.open if str(path).endswith(".gz") else open
    name, chunks = None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


class GCIndex:
    def __init__(self, fasta, names):
        """fasta: target sequences; names: target order of the libraries (every name must be in fasta)."""
        idx = {n: i for i, n in enumerate(names)}
        bits = [None] * len(names)
        lens = np.zeros(len(names), dtype=np.int64)
        for n, seq in _read_fasta(fasta):
            i = idx.get(n)
            if i is None:
                continue
            g = _GC[np.frombuffer(seq.encode(), dtype=np.uint8)]
            lens[i] = len(g)
            bits[i] = np.packbits(g)
        missing = [names[i] for i, b in enumerate(bits) if b is None]
        if missing:
            raise ValueError(f"{len(missing)} targets missing from {fasta}, e.g. {missing[:3]}")
        self.lens = lens
        self.offsets = np.r_[0, np.cumsum([len(b) for b in bits])]      # byte offsets
        self.bits = np.concatenate(bits)

    def _cum(self, t):
        """Cumulative G/C counts of target t, length L + 1."""
        g = np.unpackbits(self.bits[self.offsets[t]:self.offsets[t + 1]], count=self.lens[t])
        return np.r_[0, np.cumsum(g, dtype=np.int32)]

    def _chunks(self, ts, max_bases=50_000_000):
        """Consecutive groups of (sorted unique) targets with at most max_bases bases each."""
        start, acc = 0, 0
        for i, t in enumerate(ts):
            acc += self.lens[t]
            if acc > max_bases and i > start:
                yield ts[start:i]
                start, acc = i, self.lens[t]
        if start < len(ts):
            yield ts[start:]

    def _flat_cum(self, ts):
        """Concatenated cumulative counts of targets ts (each L + 1 long) and their start offsets."""
        cums = [self._cum(t) for t in ts]
        off = np.r_[0, np.cumsum([len(c) for c in cums])]
        return np.concatenate(cums), off

    def fragment_gc(self, tx, start, flen):
        tx, start, flen = (np.asarray(a, dtype=np.int64) for a in (tx, start, flen))
        out = np.full(len(tx), np.nan)
        order = np.argsort(tx, kind="stable")
        ts = np.unique(tx)
        for chunk in self._chunks(ts):
            cum, off = self._flat_cum(chunk)
            sel = order[np.searchsorted(tx[order], chunk[0]):np.searchsorted(tx[order], chunk[-1], side="right")]
            pos = off[np.searchsorted(chunk, tx[sel])]
            L = self.lens[tx[sel]]
            s = np.clip(start[sel], 0, L)
            e = np.clip(start[sel] + flen[sel], s + 1, L)
            n = np.maximum(e - s, 1)
            out[sel] = (cum[pos + e] - cum[pos + s]) / n
        return out

    def position_hist(self, lengths, edges, max_pos=128):
        """(T, len(lengths), K) float32 share of start positions per G/C bin (uniform where L < length)."""
        lengths = np.asarray(lengths, dtype=np.int64)
        K = len(edges) - 1
        T = len(self.lens)
        H = np.zeros((T, len(lengths), K), dtype=np.float32)
        for chunk in self._chunks(np.arange(T)):
            cum, off = self._flat_cum(chunk)
            L = self.lens[chunk]
            for j, f in enumerate(lengths):
                npos = np.where(L >= f, np.minimum(max_pos, L - f + 1), 0)
                tot = npos.sum()
                if tot == 0:
                    continue
                rows = np.repeat(np.arange(len(chunk)), npos)
                k = np.arange(tot) - np.repeat(np.cumsum(npos) - npos, npos)          # 0..npos-1 per target
                span = np.repeat(L - f, npos)
                s = np.where(np.repeat(npos, npos) > 1, (k * span) // np.maximum(np.repeat(npos, npos) - 1, 1), 0)
                base = np.repeat(off[:-1], npos)
                gc = (cum[base + s + f] - cum[base + s]) / f
                b = np.clip(np.searchsorted(edges, gc, side="right") - 1, 0, K - 1)
                h = np.bincount(rows * K + b, minlength=len(chunk) * K).reshape(len(chunk), K).astype(np.float32)
                h /= np.maximum(h.sum(axis=1, keepdims=True), 1.0)
                H[chunk, j] = h
            short = L < lengths.min()
            H[chunk[short]] = 1.0 / K
        empty = H.sum(axis=2) == 0
        H[empty] = 1.0 / K
        return H


GC_EDGES = np.r_[0.0, np.arange(0.25, 0.751, 0.05), 1.0]          # 12 bins, outer bins open-ended
GC_LENGTHS = np.array([100, 150, 200, 250, 300, 400, 500, 700])


class GCTable:
    """
    Position G/C histograms shared by the FragmentModels of a reference: H (T, len(lengths), K) from
    GCIndex.position_hist, and the index for fragment G/C (not pickled; only needed to compute features).
    """

    def __init__(self, index, lengths=GC_LENGTHS, edges=GC_EDGES, max_pos=128):
        self.index = index
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.H = index.position_hist(lengths, edges, max_pos=max_pos)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "index"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.index = None

    def fragment_bins(self, tx, start, flen):
        if self.index is None:
            raise ValueError("GCTable without a sequence index (unpickled): fragment G/C needs GCIndex")
        g = self.index.fragment_gc(tx, start, flen)
        return np.clip(np.searchsorted(self.edges, g, side="right") - 1, 0, len(self.edges) - 2)

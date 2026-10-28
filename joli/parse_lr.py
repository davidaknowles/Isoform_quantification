"""
parse_lr.py
===========
Cell barcode, UMI and RT-primer calling for long reads of Parse Evercode WT
libraries (e.g. PacBio Kinnex S-reads after array segmentation).

Read in the orientation Illumina read 2 sees (the "antisense" orientation),
a read is

    adapter | UMI(10) | bc3(8) | linker A(12) | bc2(8) | linker B(12) | bc1(8) |
    poly(T) or random-hexamer site | cDNA (antisense) | template-switch oligo (rc)

with the WT v3 linkers (read-2 positions 19-30 and 39-50). The cell barcode
is bc3+bc2+bc1 (read-2 order, as salmon's split-seq geometry concatenates
them); bc1 also identifies the RT primer, poly(dT) or random hexamer.
"""

from collections import Counter
from dataclasses import dataclass

COMP = str.maketrans("ACGTN", "TGCAN")
LINKER_A = "ATGAGGGGTCAG"
LINKER_B = "TCCAACCACCTC"
TSO = "AAGCAGTGGTATCAACGCAGAGTGAAT"     # followed by a few G (template switch)
ADAPTER_END = "CAGACGTGTGCTCTTCCGATCT"  # 3' end of the adapter preceding the UMI


def revcomp(s):
    return s.translate(COMP)[::-1]


def _hamming_le1(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1


def find_near(seq, pattern, expected, slack=3):
    """Position of pattern in seq within expected +- slack, exact first, then <= 1 mismatch."""
    lo, hi = max(0, expected - slack), min(len(seq) - len(pattern), expected + slack)
    k = seq.find(pattern, lo, hi + len(pattern))
    if k >= 0:
        return k
    for k in range(lo, hi + 1):
        if _hamming_le1(seq[k:k + len(pattern)], pattern):
            return k
    return -1


class Corrector:
    """Map a barcode to its whitelist entry exactly or within one mismatch (when unique)."""

    def __init__(self, whitelist):
        self.exact = set(whitelist)
        neigh = {}
        for w in whitelist:
            for i, c in enumerate(w):
                for b in "ACGT":
                    if b != c:
                        neigh.setdefault(w[:i] + b + w[i + 1:], set()).add(w)
        self.one = {k: next(iter(v)) for k, v in neigh.items() if len(v) == 1 and k not in self.exact}

    def __call__(self, bc):
        if bc in self.exact:
            return bc, 0
        w = self.one.get(bc)
        return (w, 1) if w is not None else (None, -1)


@dataclass
class ParseRead:
    umi: str
    bc3: str
    bc2: str
    bc1: str
    primer: str
    cdna: str          # sense orientation, TSO and poly(A) removed
    polya: int         # length of the removed poly(A) run
    tso: bool          # template-switch oligo found at the 5' end
    n_corrected: int   # barcode segments corrected by one mismatch


class ParseReadParser:
    """
    Args:
        whitelists : (bc3 list, bc2 list, bc1 list), read-2 segments 1-3.
        bc1_primer : dict bc1 -> "poly(dT)" | "random hexamer".
        linker_a_pos : expected start of linker A in the antisense read.
    """

    def __init__(self, whitelists, bc1_primer, linker_a_pos=62, slack=3, min_cdna=50):
        self.corr = [Corrector(w) for w in whitelists]
        self.bc1_primer = bc1_primer
        self.a_pos, self.slack, self.min_cdna = linker_a_pos, slack, min_cdna

    def orient(self, seq):
        """Antisense orientation (adapter and barcodes first) or None."""
        head = seq[:self.a_pos + 40]
        if find_near(head, LINKER_A, self.a_pos, self.slack) >= 0:
            return seq
        rc = revcomp(seq)
        if find_near(rc[:self.a_pos + 40], LINKER_A, self.a_pos, self.slack) >= 0:
            return rc
        return None

    def parse(self, seq):
        """ParseRead, or (None, reason) if the read cannot be assigned."""
        s = self.orient(seq)
        if s is None:
            return None, "no_linker_a"
        a = find_near(s, LINKER_A, self.a_pos, self.slack)
        b = find_near(s, LINKER_B, a + 20, 2)
        if b < 0:
            return None, "no_linker_b"
        if a < 18:
            return None, "short_umi"
        raw = (s[a - 8:a], s[a + 12:b], s[b + 12:b + 20])
        if len(raw[1]) != 8:
            return None, "bc2_length"
        fixed, n_corr = [], 0
        for c, x in zip(self.corr, raw):
            w, d = c(x)
            if w is None:
                return None, "barcode"
            fixed.append(w)
            n_corr += d
        primer = self.bc1_primer.get(fixed[2])
        if primer is None:
            return None, "bc1_primer"
        insert = s[b + 20:]
        polya = 0
        if primer == "poly(dT)":
            while polya < len(insert) and insert[polya] == "T":
                polya += 1
        cdna = revcomp(insert[polya:])
        tso = False
        k = cdna.find(TSO, 0, 40)
        if k >= 0:
            j = k + len(TSO)
            while j < len(cdna) and cdna[j] == "G":
                j += 1
            cdna, tso = cdna[j:], True
        if len(cdna) < self.min_cdna:
            return None, "short_cdna"
        return ParseRead(umi=s[a - 18:a - 8], bc3=fixed[0], bc2=fixed[1], bc1=fixed[2], primer=primer,
                         cdna=cdna, polya=polya, tso=tso, n_corrected=n_corr), "ok"


def primer_pairs_from_cells(cells):
    """
    Pair poly(dT) and random-hexamer bc1 barcodes (the two RT primers of one
    well) from labelled cells that list both halves.

    Args:
        cells : iterable of (barcode, primer) with barcode = bc3+bc2+bc1 (8+8+8).
    Returns dict poly(dT) bc1 -> random-hexamer bc1, taking for each poly(dT)
    barcode the hexamer barcode it shares the most bc3+bc2 prefixes with (true
    partners co-occur in every cell, others only where prefixes collide).
    """
    by_prefix = {}
    for bc, primer in cells:
        by_prefix.setdefault(bc[:16], {}).setdefault(primer, set()).add(bc[16:])
    co = Counter()
    for d in by_prefix.values():
        for dt in d.get("poly(dT)", ()):
            for hx in d.get("random hexamer", ()):
                co[(dt, hx)] += 1
    best = {}
    for (dt, hx), n in co.items():
        if n > best.get(dt, (None, 0))[1]:
            best[dt] = (hx, n)
    return {dt: hx for dt, (hx, _) in best.items()}

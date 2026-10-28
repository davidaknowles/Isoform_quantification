"""Tests for Parse long-read barcode calling."""

from joli.parse_lr import LINKER_A, LINKER_B, TSO, ParseReadParser, primer_pairs_from_cells, revcomp

ADAPTER = "CTACACGACGCTCTTCCGATCTCAGACGTGTGCTCTTCCGATCT"   # 44 nt, linker A at 44 + 18 = 62
WL = (["AAAAAAAA", "CCCCCCCC"], ["GGGGGGGG", "TTTTTTTT"], ["ACGTACGT", "TGCATGCA"])
PRIMER = {"ACGTACGT": "poly(dT)", "TGCATGCA": "random hexamer"}
CDNA = "ATGGCTAGCTAGGATCGATCGTAGCTAGCTAGCTGATCGATCGATGCTAGCTAGCTAGCATCGATCGATCGTAGCTAGC"


def antisense_read(umi, bc3, bc2, bc1, la=LINKER_A, polyt=15):
    return ADAPTER + umi + bc3 + la + bc2 + LINKER_B + bc1 + "T" * polyt + revcomp(TSO + "GGG" + CDNA)


def test_parse_both_orientations_and_corrections():
    p = ParseReadParser(WL, PRIMER)
    read = antisense_read("ACGTACGTAC", "AAAAAAAA", "TTTTTTTT", "ACGTACGT")
    for seq in (read, revcomp(read)):
        r, why = p.parse(seq)
        assert why == "ok"
        assert (r.umi, r.bc3, r.bc2, r.bc1, r.primer) == ("ACGTACGTAC", "AAAAAAAA", "TTTTTTTT", "ACGTACGT", "poly(dT)")
        assert r.cdna == CDNA and r.polya == 15 and r.tso and r.n_corrected == 0
    # one mismatch in linker A and in bc3
    bad_la = LINKER_A[:5] + ("A" if LINKER_A[5] != "A" else "C") + LINKER_A[6:]
    r, why = p.parse(antisense_read("ACGTACGTAC", "AAAAAAAC", "GGGGGGGG", "TGCATGCA", la=bad_la, polyt=0))
    assert why == "ok" and r.bc3 == "AAAAAAAA" and r.n_corrected == 1 and r.primer == "random hexamer"


def test_primer_pairs():
    cells = [("AAAAAAAAGGGGGGGG" + "ACGTACGT", "poly(dT)"), ("AAAAAAAAGGGGGGGG" + "TGCATGCA", "random hexamer"),
             ("CCCCCCCCGGGGGGGG" + "ACGTACGT", "poly(dT)"), ("CCCCCCCCGGGGGGGG" + "TGCATGCA", "random hexamer")]
    assert primer_pairs_from_cells(cells) == {"ACGTACGT": "TGCATGCA"}

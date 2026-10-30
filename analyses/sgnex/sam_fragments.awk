# salmon --writeMappings SAM (paired-end) -> one line per fragment alignment:
#   fragment id, target index (@SQ order, 0-based), leftmost start (0-based), fragment length, AS of both mates
# Only proper pairs (flag 0x2); mates of one alignment are consecutive lines with the same QNAME and HI.
# @SQ names are written to the file given by -v refs=<path>.
BEGIN { OFS = "\t"; nsq = 0; rid = -1; prevq = "" }
/^@SQ/ { split($2, a, ":"); idx[a[2]] = nsq++; print a[2] > refs; next }
/^@/ { next }
{
    if (and($2, 2) == 0) next
    hi = ""; as = 0
    for (i = 12; i <= NF; i++) {
        if (substr($i, 1, 5) == "HI:i:") hi = substr($i, 6)
        else if (substr($i, 1, 5) == "AS:i:") as = substr($i, 6) + 0
    }
    if ($1 != prevq) { rid++; prevq = $1; have = 0 }
    key = $3 SUBSEP hi
    if (have && key == pkey) {
        start = ($4 < ppos ? $4 : ppos) - 1
        tl = $9 < 0 ? -$9 : $9
        print rid, idx[$3], start, tl, as + pas
        have = 0
    } else { pkey = key; ppos = $4; pas = as; have = 1 }
}

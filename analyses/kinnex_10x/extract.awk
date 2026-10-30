# PacBio scisoseq dedup BAM (samtools view) -> chunks of cDNA fasta and read tables.
# One read per molecule, oriented 5'->3', adapters and poly(A) removed by PacBio.
# Vars: lib (library id), lib_index (int), kit (3p/5p), out (dir), chunk (reads per chunk).
# Read ids are lib_index * 1e10 + n so they are unique across libraries.
BEGIN { OFS = "\t"; n = 0; k = -1 }
{
    if (n % chunk == 0) {
        if (k >= 0) { close(fa); close(tsv) }
        k++
        fa = sprintf("gzip -1 > %s/cdna_%s_%03d.fa.gz", out, lib, k)
        tsv = sprintf("gzip -1 > %s/reads_%s_%03d.tsv.gz", out, lib, k)
        print "read", "cb", "umi", "primer", "kit", "cdna_len" | tsv
    }
    cb = ""; um = ""
    for (i = 12; i <= NF; i++) {
        if (substr($i, 1, 5) == "CB:Z:") cb = substr($i, 6)
        else if (substr($i, 1, 5) == "XM:Z:") um = substr($i, 6)
    }
    id = sprintf("%d%010d", lib_index, n)
    print ">" id "\n" $10 | fa
    print id, lib "_" cb, um, "poly(dT)", kit, length($10) | tsv
    n++
}

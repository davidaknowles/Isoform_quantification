"""
Write a complete GTF (gene, transcript and exon records) for IsoQuant from the SG-NEx annotation.

The SG-NEx v5_reformatted GTF repeats a transcript record for every exon, and the "corrected" GTF
(same exons) has one transcript record per transcript but no gene records. This adds gene records
spanning each gene's transcripts to the corrected GTF. A gene id seen on more than one
chromosome/strand gets a suffix (_2, _3, ...) for the later ones, so every gene is one locus;
a gene id equal to a transcript id gets the suffix _gene. Transcript ids are unchanged.

Run: python analyses/baselines/make_complete_gtf.py <corrected.gtf> <out.gtf>
"""

import re
import sys


def main(gtf_in, gtf_out):
    gid_re = re.compile(r'gene_id "([^"]*)"')
    rows, loci, n_loci, span = [], {}, {}, {}
    with open(gtf_in) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.rstrip("\n").split("\t")
            g = gid_re.search(v[8]).group(1)
            key = (g, v[0], v[6])
            if key not in loci:
                n_loci[g] = n_loci.get(g, 0) + 1
                loci[key] = g if n_loci[g] == 1 else f"{g}_{n_loci[g]}"
            gl = loci[key]
            if gl != g:
                v[8] = v[8].replace(f'gene_id "{g}"', f'gene_id "{gl}"')
            s, e = int(v[3]), int(v[4])
            a, b = span.get(gl, (s, e))
            span[gl] = (min(a, s), max(b, e))
            rows.append(v)
    # IsoQuant rejects a gene id equal to a transcript id (single-isoform long SIRVs)
    tid_re = re.compile(r'transcript_id "([^"]*)"')
    tids = {m.group(1) for v in rows for m in [tid_re.search(v[8])] if m}
    clash = {g for g in span if g in tids}
    for v in rows:
        gl = gid_re.search(v[8]).group(1)
        if gl in clash:
            v[8] = v[8].replace(f'gene_id "{gl}"', f'gene_id "{gl}_gene"')
    span.update({f"{g}_gene": span.pop(g) for g in clash})
    with open(gtf_out, "w") as f:
        done = set()
        for v in rows:
            gl = gid_re.search(v[8]).group(1)
            if gl not in done:
                a, b = span[gl]
                f.write("\t".join([v[0], v[1], "gene", str(a), str(b), ".", v[6], ".", f'gene_id "{gl}";']) + "\n")
                done.add(gl)
            f.write("\t".join(v) + "\n")
    print(f"{len(done)} genes, {len(rows)} records, {sum(1 for k, g in loci.items() if g != k[0])} renamed loci, "
          f"{len(clash)} genes renamed <gene>_gene")


if __name__ == "__main__":
    main(*sys.argv[1:3])

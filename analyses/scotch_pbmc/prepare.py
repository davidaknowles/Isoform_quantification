"""
SCOTCH human PBMC (Xu et al.), Parse Evercode WT Mini v2: the public reads were demultiplexed by Parse split-pipe, and
each read name carries <bc1 well>_<bc2 well>_<bc3 well>__<primer T/R>__<raw wells>__<bc1>_<bc2>_<bc3>__<UMI>, with bc1
the merged (oligo-dT) well. Illumina cDNA reads have no barcode read, so for salmon/alevin-fry this writes a synthetic
barcode read per cDNA read, in input order: bc1 + bc2 + bc3 (24 nt) + UMI (10 nt). It also writes the round-1 table
(barcode, sample description, parse barcode type, well) used by igvf_cortex/sr_cells.py and ont_barcodes.py, with the
merged well as the well, so a cell is <bc1 well>_<bc2 index>_<bc3 index> on both platforms.

Only Sample8 is usable: Sample7's Illumina reads (SRR31630767) carry no barcodes in their names.

Run: python analyses/scotch_pbmc/prepare.py <scotch_dir> <sample> <illumina fastq.gz>...
     writes <scotch_dir>/<sample>/barcodes.fastq.gz and bc1_map.tsv
"""

import gzip
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def main(sdir, sample, *fastqs):
    out = Path(sdir) / sample
    out.mkdir(parents=True, exist_ok=True)
    bc1 = Counter()
    n = 0
    with gzip.open(out / "barcodes.fastq.gz", "wt", compresslevel=1) as fo:
        for f in fastqs:
            with gzip.open(f, "rt") as fh:
                for i, line in enumerate(fh):
                    if i % 4 != 0:
                        continue
                    name = line.split()
                    fields = name[1].split("__")
                    wells, primer, seqs, umi = fields[0], fields[1], fields[3].split("_"), fields[4][:10]
                    bc1[(seqs[0], primer, int(wells.split("_")[0]))] += 1
                    seq = "".join(seqs) + umi
                    fo.write(f"{name[0]}\n{seq}\n+\n{'I' * len(seq)}\n")
                    n += 1
    m = pd.DataFrame([{"barcode": b, "sample description": sample, "parse barcode type": p, "well": f"W{w:02d}", "reads": c}
                      for (b, p, w), c in bc1.items()])
    m = m[m.reads >= 0.001 * m.reads.sum()].sort_values(["well", "parse barcode type"])
    m.to_csv(out / "bc1_map.tsv", sep="\t", index=False)
    print(f"{n} reads; round-1 table:\n{m.to_string(index=False)}")


if __name__ == "__main__":
    main(*sys.argv[1:])

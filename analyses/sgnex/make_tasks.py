"""
Task table for analyses/sgnex/align.sh: one row per (cell line, replicate,
protocol) with its fastq files (runs of the same replicate merged).

Run: python analyses/sgnex/make_tasks.py <download_list.txt> <raw_dir> <out_tasks.tsv>
"""

import re
import sys
from pathlib import Path

import pandas as pd


def main(url_list, raw_dir, out):
    rows = []
    for u in Path(url_list).read_text().split():
        f = Path(u).name
        m = re.match(r"SGNex_(\w+?)_(Illumina|directRNA|directcDNA|cDNA|PacBio-SMRTcell)_replicate(\d+)_run(\d+)(_R[12])?\.fastq\.gz", f)
        if m:
            cell, proto, rep, run, mate = m.groups()
            rows.append((cell, int(rep), proto, mate or "", str(Path(raw_dir) / f)))
    df = pd.DataFrame(rows, columns=["cell", "rep", "protocol", "mate", "path"])
    out_rows = []
    for (cell, rep, proto), g in df.groupby(["cell", "rep", "protocol"]):
        if proto == "Illumina":
            r1 = g[g.mate == "_R1"].path.tolist(); r2 = g[g.mate == "_R2"].path.tolist()
            out_rows.append((cell, rep, proto, ",".join(r1), ",".join(r2)))
        else:
            out_rows.append((cell, rep, proto, ",".join(sorted(g.path)), ""))
    t = pd.DataFrame(out_rows, columns=["cell", "rep", "protocol", "reads1", "reads2"])
    t.to_csv(out, sep="\t", index=False)
    print(t[["cell", "rep", "protocol"]].to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])

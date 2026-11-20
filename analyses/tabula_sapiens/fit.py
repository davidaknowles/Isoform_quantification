"""
Tabula Sapiens atlas pilot fits: the mouse-brain fit (analyses/mouse_brain/fit.py: SR gene profiles, LR_bulk, joint
design_gene with bulk long-read libraries as mixtures over their tissue's cell classes) with human cell
classes and lineages. Region = tissue (Muscle, Heart); class effects are shared across tissues, so e.g.
endothelial cells and macrophages of heart and muscle borrow from each other.

Run: python analyses/tabula_sapiens/fit.py <libraries.pkl> <out_dir> [ct_sd] [composition.tsv | -] [key=value ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_brain"))
import fit as F  # noqa: E402

F.LINEAGE = {"mesenchymal stem cell": "stromal", "fibroblast": "stromal", "satellite cell": "muscle",
             "fast muscle cell": "muscle", "slow muscle cell": "muscle", "cardiac muscle cell": "muscle",
             "smooth muscle cell": "mural", "pericyte": "mural", "endothelial": "vascular",
             "lymphatic endothelial": "vascular", "T cell": "immune", "macrophage": "immune",
             "mast cell": "immune", "hepatocyte-labelled": "other"}

if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if "=" not in a]
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    F.main(*pos[:2], *(float(x) for x in pos[2:3]), *pos[3:4], **kw)

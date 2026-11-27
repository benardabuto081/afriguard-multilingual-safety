#!/usr/bin/env python3
"""Official-format submission zip from saved label log-probs (argmax of summed log-prob) for one template."""
import argparse
import zipfile
from pathlib import Path

import numpy as np

from afriguard.labels import LANGUAGES, TASK_A_LABELS, TASK_B_LABELS, TASK_DIRS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
z = np.load(Path(a.run) / f"lp_{a.template}.npz")
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
    for lang in LANGUAGES:
        for task, labs in (("A", TASK_A_LABELS), ("B", TASK_B_LABELS)):
            ids, S = z[f"{lang}_{task}_ids"], z[f"{lang}_{task}"]
            body = "id,label\n" + "".join(f'{i},"{labs[k]}"\n' for i, k in zip(ids, S.argmax(1)))
            zf.writestr(f"{TASK_DIRS[task]}/{lang}.csv", body)
print("wrote", a.out)

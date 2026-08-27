"""Decoupled workflow, step 2 of 3: EXECUTE locally.

Runs run_one over every manifest entry (or the given names) in the
prepared RELAX tree. This is the Mac-local twin of the SLURM array
template that prepare.py writes; both drive the identical run_one.py +
job.json, so results are the same wherever they ran.

Usage: python -m hull0k.execute configs/Ni-Cu.yaml <potential-key> \
           [--jobs=N] [names...]

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .build_pool import load_config
from . import run_one


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    jobs = 1
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    cfg = load_config(args[0])
    potkey = args[1]
    relax_dir = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"] / "RELAX"
    manifest = (relax_dir / "manifest.txt").read_text().split()
    names = args[2:] or manifest
    rcs = {}
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one.main, relax_dir / n): n for n in names}
            for fut, n in futs.items():
                rcs[n] = fut.result()
    else:
        for n in names:
            rcs[n] = run_one.main(relax_dir / n)
    failed = [n for n, rc in rcs.items() if rc != 0]
    print(f"executed {len(names)} structures, {len(failed)} failed"
          + (f": {failed}" if failed else ""))
    print("next: python -m hull0k.harvest " + " ".join(args[:2]))


if __name__ == "__main__":
    main(sys.argv[1:])

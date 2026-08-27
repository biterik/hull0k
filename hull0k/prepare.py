"""Decoupled workflow, step 1 of 3: PREPARE (no LAMMPS needed).

Renders every LAMMPS input for every candidate into
<data_tree>/<thread_dir>/RELAX/<name>/ together with a job.json each,
plus tree-level artifacts that make the RELAX directory SELF-CONTAINED
for execution anywhere (Mac or a cluster compute node with any plain
python3):

  RELAX/manifest.txt            one structure name per line
  RELAX/run_one.py              stdlib-only per-structure driver (copy)
  RELAX/run-all-local.sh        serial local loop (bash)
  RELAX/submit-hull0k-array.sh  SLURM array TEMPLATE -- every cluster-
                                specific value is an EDITME_ token that
                                MUST be filled in before submitting
                                (partition, module, walltime, lmp path)
  RELAX/README-EXECUTE.md       how to run locally or on a cluster

Pipeline:  prepare -> execute (or sbatch) -> harvest
The harvest step (hull0k.harvest) writes relax-summary_<pot>.yaml INTO
the thread directory next to the results, wherever that tree lives.

Usage: python -m hull0k.prepare configs/Ni-Cu.yaml <potential-key> [names...]

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import json
import shutil
import sys
from pathlib import Path

import yaml

from .build_pool import load_config
from .relax import (REPO, MAX_FALLBACK_CYCLES, FTOL_PER_DOF, PRESS_TOL_BAR,
                    pair_lines, press_gate_components, render)


def prepare_one(cfg, potkey, name, relax_dir):
    pot = cfg["potentials"][potkey]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    rundir = relax_dir / name
    rundir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    with open(pool / f"{name}.yaml") as f:
        side = yaml.safe_load(f)
    coupling = side["box_coupling"]
    natoms = side["natoms"]
    shutil.copy2(pool / f"{name}.lmps", rundir / f"{name}.lmps")

    maxiter = max(2000, 20 * natoms)
    style_line, coeff_line = pair_lines(pot)
    label = f"{name}_{potkey}"
    boxprep = "change_box all triclinic" if coupling == "tri" else ""
    common = {"THREADDIR": pot["thread_dir"], "RUNDATE": today,
              "PAIRSTYLELINE": style_line, "PAIRCOEFFLINE": coeff_line,
              "MAXITER": maxiter, "MAXEVAL": 2 * maxiter}

    def boxrelax_stage(lab, datafile):
        infile = f"{lab}-min.in"
        (rundir / infile).write_text(render(
            REPO / "templates/minimize-boxrelax.in.skel",
            dict(common, RUNLABEL=lab, DATAFILE=datafile,
                 COUPLING=coupling, BOXPREP=boxprep)))
        return {"label": lab, "in": infile, "log": f"{lab}-min.log",
                "out": f"{lab}-relaxed.lmps"}

    def fixedcell_stage(lab, datafile):
        infile = f"{lab}-fixmin.in"
        (rundir / infile).write_text(render(
            REPO / "templates/minimize-fixedcell.in.skel",
            dict(common, RUNLABEL=lab, DATAFILE=datafile)))
        return {"label": lab, "in": infile, "log": f"{lab}-fixmin.log",
                "out": f"{lab}-fixmin-relaxed.lmps"}

    stage0 = boxrelax_stage(label, f"{name}.lmps")
    cycles = []
    prev_box_out = f"{name}.lmps"
    for cyc in range(MAX_FALLBACK_CYCLES):
        fix = fixedcell_stage(f"{label}-c{cyc}a", prev_box_out)
        box = boxrelax_stage(f"{label}-c{cyc}b", fix["out"])
        cycles.append([fix, box])
        prev_box_out = box["out"]

    final_name = f"{label}-final-relaxed.lmps"
    splab = f"{label}-equiv"
    spin = f"{splab}-singlepoint.in"
    (rundir / spin).write_text(render(
        REPO / "templates/singlepoint.in.skel",
        dict(common, RUNLABEL=splab, DATAFILE=final_name)))

    launch = (cfg.get("lmp_launch") or "").split()
    job = {
        "name": name, "label": label, "natoms": natoms, "coupling": coupling,
        "lmp_cmd": launch + [cfg["lmp_binary"]],
        "ftol_per_dof": FTOL_PER_DOF, "press_tol_bar": PRESS_TOL_BAR,
        # from relax.py, so the staged tree gates on exactly what the
        # integrated runner would have gated on
        "press_gate_components": press_gate_components(coupling),
        "stage0": stage0, "cycles": cycles,
        "final_name": final_name,
        "singlepoint": {"label": splab, "in": spin,
                        "log": f"{splab}-singlepoint.log", "out": None},
        "timeout_s": 1200,
        "prepared": today, "potential": potkey,
    }
    with open(rundir / "job.json", "w") as f:
        json.dump(job, f, indent=1)
    return name


SBATCH_TEMPLATE = """#!/bin/bash
# submit-hull0k-array.sh -- SLURM array over the hull0k RELAX manifest.
# TEMPLATE: fill in every EDITME_ token for the target cluster BEFORE
# submitting (partition, module, walltime; set lmp_binary/lmp_launch in
# the config BEFORE `prepare` so job.json carries the cluster lmp).
# One array task = one structure = one run directory.
# Author: Erik Bitzek (created with LLM-LMPS)  Date: {today}
#SBATCH --job-name=hull0k-relax
#SBATCH --partition=EDITME_PARTITION
#SBATCH --time=EDITME_WALLTIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{ntasks}
#SBATCH -o hull0k-relax-%A_%a.out
set -euo pipefail
module purge
EDITME_MODULE_LOAD_LINE

cd "$SLURM_SUBMIT_DIR"
name=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" manifest.txt)
[[ -n "$name" ]] || {{ echo "no manifest entry for task $SLURM_ARRAY_TASK_ID"; exit 1; }}
[[ -f "$name/job.json" ]] || {{ echo "missing $name/job.json"; exit 1; }}
python3 run_one.py "$name"
"""

README_EXECUTE = """# README-EXECUTE.md -- running the prepared hull0k RELAX tree

This directory is SELF-CONTAINED for execution: every run dir holds all
pre-rendered LAMMPS inputs and a job.json; run_one.py needs only plain
python3 (no venv) and the lmp binary named inside job.json.

## Locally (Mac)

    python -m hull0k.execute <config> <potential-key> [--jobs N]

or, without the repo/venv:

    bash run-all-local.sh

## On a cluster (manual, canon-compliant)

1. BEFORE preparing: set `lmp_binary` (cluster path) and, if the binary
   is MPI-parallel, `lmp_launch: "srun"` in the config, then re-run
   prepare so job.json carries the cluster command.
2. Stage this RELAX directory (and the potential files at the paths
   job.json references) to cluster scratch (PTMP, never $HOME) with
   rsync. LAMMPS runs ONLY inside a SLURM job, never on a login node.
3. Edit every EDITME_ token in submit-hull0k-array.sh, then sbatch it
   from inside this directory. Probe rule: submit a 1-task array first
   (--array=1-1), read its output, then release the rest.
4. Pull the tree back to the Mac (rsync) and run the harvest there:

    python -m hull0k.harvest <config> <potential-key>

The harvest writes relax-summary_<potential>.yaml into the thread
directory next to these results; hull/report consume it unchanged.
"""


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    names = argv[2:] or sorted(y.stem for y in pool.glob("*.yaml"))
    relax_dir = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"] / "RELAX"
    relax_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        prepare_one(cfg, potkey, name, relax_dir)
        print(f"  prepared {name}")
    (relax_dir / "manifest.txt").write_text("\n".join(names) + "\n")
    shutil.copy2(Path(__file__).parent / "run_one.py", relax_dir / "run_one.py")
    (relax_dir / "run-all-local.sh").write_text(
        "#!/bin/bash\n# serial local execution of the hull0k manifest\n"
        "set -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
        "while read -r name; do python3 run_one.py \"$name\"; done < manifest.txt\n")
    today = datetime.date.today().isoformat()
    (relax_dir / "submit-hull0k-array.sh").write_text(
        SBATCH_TEMPLATE.format(today=today, ntasks=len(names)))
    (relax_dir / "README-EXECUTE.md").write_text(README_EXECUTE)
    print(f"prepared {len(names)} structures in {relax_dir}")
    print("next: python -m hull0k.execute (local) or stage + sbatch (cluster), "
          "then python -m hull0k.harvest")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Per-structure execution driver for the decoupled relax workflow.

STDLIB-ONLY by design: this file is copied into the staged RELAX tree by
prepare.py and must run with any plain python3 (cluster compute node, no
venv). It executes the pre-rendered LAMMPS inputs for ONE structure:

  1. one-shot box-relax stage;
  2. if unconverged (or pressure gate fails): the two-stage fallback,
     alternating pre-rendered fixed-cell / box-relax inputs;
  3. copies the winning relaxed file to <label>-final-relaxed.lmps and
     runs the pre-rendered equivalence single-point against it.

Everything it needs (file names, gates, the full lmp command) comes from
job.json in the run directory, written by prepare.py. It writes
result-status.json; all venv-dependent checks (spglib, ase) happen later
in harvest.py on the Mac.

Usage: python3 run_one.py <run-directory>

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_stage(job, rundir, stage):
    cmd = list(job["lmp_cmd"]) + ["-in", stage["in"], "-screen", "none"]
    r = subprocess.run(cmd, cwd=rundir, capture_output=True, text=True,
                       timeout=job.get("timeout_s", 1200))
    return r.returncode


def parse_log(rundir, stage):
    out = {}
    stop = None
    label = stage["label"]
    for line in (rundir / stage["log"]).read_text().splitlines():
        m = re.match(rf"RESULT {re.escape(label)} (\S+) = (\S+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
        if line.startswith("Stopping criterion"):
            stop = line.split("=", 1)[1].strip()
    out["stopping_criterion"] = stop
    return out


def converged(job, res):
    s = res.get("stopping_criterion") or ""
    return ("force tolerance" in s.lower()
            or res.get("fnorm_per_dof_eV_A", 1.0) < job["ftol_per_dof"])


def press_ok(job, res):
    return all(res.get(f"{k}_bar") is not None
               and abs(res[f"{k}_bar"]) < job["press_tol_bar"]
               for k in job["press_gate_components"])


def main(rundir):
    rundir = Path(rundir)
    with open(rundir / "job.json") as f:
        job = json.load(f)

    status = {"name": job["name"], "label": job["label"], "path": "boxrelax",
              "cycles_used": 0}
    rc = run_stage(job, rundir, job["stage0"])
    res = parse_log(rundir, job["stage0"]) if rc == 0 else {}
    winner = job["stage0"]
    ok = rc == 0 and converged(job, res) and press_ok(job, res)

    if not ok:
        status["path"] = "twostage-fallback"
        for i, (fix, box) in enumerate(job["cycles"]):
            status["cycles_used"] = i + 1
            rc = run_stage(job, rundir, fix)
            if rc != 0:
                break
            rc = run_stage(job, rundir, box)
            if rc != 0:
                break
            res = parse_log(rundir, box)
            winner = box
            if converged(job, res) and press_ok(job, res):
                ok = True
                break
        if rc != 0 or not res:
            status["status"] = "FAILED"
            with open(rundir / "result-status.json", "w") as f:
                json.dump(status, f, indent=1)
            return 1
        if not ok:
            status["path"] += "-UNCONVERGED"

    shutil.copy2(rundir / winner["out"], rundir / job["final_name"])
    status["winner_stage"] = winner["label"]
    status["winner_out"] = winner["out"]
    status["results"] = res

    sp = job["singlepoint"]
    rc_sp = run_stage(job, rundir, sp)
    status["singlepoint"] = parse_log(rundir, sp) if rc_sp == 0 else None
    status["status"] = "OK"
    with open(rundir / "result-status.json", "w") as f:
        json.dump(status, f, indent=1)
    print(f"{job['name']}: {status['path']} E_total = {res.get('E_total_eV')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

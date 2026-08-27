"""Decoupled workflow, step 3 of 3: HARVEST (needs the venv, runs on the Mac).

Walks the executed RELAX tree (local run or pulled back from a cluster),
applies every venv-dependent integrity check -- spglib symmetry before/
after, minimum interatomic distance, the pressure gate per coupling, the
equivalence-gate comparison -- writes run.yaml per structure and merges
one summary line per structure into relax-summary_<pot>.yaml IN THE
THREAD DIRECTORY, next to the results. The summary schema is identical
to the integrated runner's (hull0k.relax), so hull/report consume either
without knowing where the compute happened.

Usage: python -m hull0k.harvest configs/Ni-Cu.yaml <potential-key> [names...]

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .build_pool import load_config
from .lmps_io import spacegroup_of
from .relax import (EQUIV_TOL_EV_PER_ATOM, MIN_DIST_A, read_lmps)


def harvest_one(cfg, potkey, name, relax_dir):
    pot = cfg["potentials"][potkey]
    elements = cfg["elements"]
    rundir = relax_dir / name
    sfile = rundir / "result-status.json"
    if not sfile.exists():
        return {"name": name, "potential": potkey, "status": "NOT-EXECUTED"}
    with open(sfile) as f:
        st = json.load(f)
    if st.get("status") != "OK":
        return {"name": name, "potential": potkey,
                "status": st.get("status", "FAILED"), "path": st.get("path")}
    with open(rundir / "job.json") as f:
        job = json.load(f)
    res = st["results"]
    natoms = job["natoms"]

    at_in = read_lmps(rundir / f"{name}.lmps", elements)
    at_out = read_lmps(rundir / st["winner_out"], elements)
    sg_before = spacegroup_of(at_in)
    sg_after = spacegroup_of(at_out)
    dmin = float(np.min(at_out.get_all_distances(mic=True)[np.triu_indices(len(at_out), 1)])) \
        if len(at_out) > 1 else np.inf
    pcomps = {k: res.get(f"{k}_bar") for k in ("press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")}
    press_ok = all(res.get(f"{k}_bar") is not None
                   and abs(res[f"{k}_bar"]) < job["press_tol_bar"]
                   for k in job["press_gate_components"])
    shear_residual = {k: pcomps.get(k) for k in ("pxy", "pxz", "pyz")
                      if k not in job["press_gate_components"]}
    equiv = None
    if st.get("singlepoint"):
        equiv = abs(st["singlepoint"]["E_total_eV"] - res["E_total_eV"]) / natoms

    summary = {
        "name": name, "potential": potkey, "status": "OK",
        "path": st["path"],
        "natoms": natoms,
        "E_total_eV": res["E_total_eV"],
        "E_per_atom_eV": res["E_total_eV"] / natoms,
        "fnorm_per_dof_eV_A": res.get("fnorm_per_dof_eV_A"),
        "fnorm_per_atom_eV_A": res.get("fnorm_per_atom_eV_A"),
        "press_components_bar": {k: (None if v is None else round(v, 3))
                                 for k, v in pcomps.items()},
        "press_gate_components": list(job["press_gate_components"]),
        "unrelaxed_shear_residual_bar": {k: (None if v is None else round(v, 3))
                                         for k, v in shear_residual.items()},
        "press_ok": bool(press_ok),
        "min_dist_A": round(dmin, 4),
        "min_dist_ok": bool(dmin > MIN_DIST_A),
        "spacegroup_before": sg_before,
        "spacegroup_after": sg_after,
        "symmetry_kept": sg_before == sg_after,
        "cell_relaxed": {k: res.get(f"{k}_A") for k in ("lx", "ly", "lz", "xy", "xz", "yz")},
        "equiv_gate_eV_per_atom": (None if equiv is None else float(f"{equiv:.3e}")),
        "equiv_ok": bool(equiv is not None and equiv < EQUIV_TOL_EV_PER_ATOM),
        "relaxed_file": st["winner_out"],
        "coupling": job["coupling"],
        "date": datetime.date.today().isoformat(),
        "executed_by": "run_one (decoupled workflow)",
    }
    with open(rundir / "run.yaml", "w") as f:
        yaml.safe_dump({
            "potential": {"key": potkey, "file": pot["file"], "sha256": pot["sha256"]},
            "lmp_cmd": job["lmp_cmd"],
            "lmp_version": cfg.get("lmp_version", "unrecorded -- see the run log header"),
            "template": "minimize-boxrelax.in.skel v1 (decoupled prepare/execute/harvest)",
            "result": summary,
        }, f, sort_keys=False)
    return summary


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    thread = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"]
    relax_dir = thread / "RELAX"
    manifest = (relax_dir / "manifest.txt").read_text().split()
    names = argv[2:] or manifest
    summaries = []
    for name in names:
        s = harvest_one(cfg, potkey, name, relax_dir)
        summaries.append(s)
        e = s.get("E_per_atom_eV")
        e_str = "n/a" if e is None else f"{e:.6f}"
        flag = "" if s.get("status") == "OK" else "  <-- " + s.get("status", "?")
        drift = "" if s.get("symmetry_kept", True) else "  [SYMMETRY CHANGED]"
        print(f"  {name:34s} {s.get('path', '?'):22s} E/at={e_str} "
              f"equiv={s.get('equiv_gate_eV_per_atom')}{flag}{drift}")
    outfile = thread / f"relax-summary_{potkey}.yaml"
    existing = []
    if outfile.exists():
        with open(outfile) as f:
            existing = yaml.safe_load(f) or []
        done = {s["name"] for s in summaries}
        existing = [e for e in existing if e["name"] not in done]
    with open(outfile, "w") as f:
        yaml.safe_dump(existing + summaries, f, sort_keys=False)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1:])

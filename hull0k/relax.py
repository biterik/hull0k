"""Phase 2: relaxation engine.

For each candidate in the pool and one chosen potential:
  1. render templates/minimize-boxrelax.in.skel with the candidate's
     box-coupling class, run lmp;
  2. two-stage fallback (alternate fixed-cell / box-relax) if the
     one-shot box/relax run stalls;
  3. integrity checks: fnorm normalizations, symmetry before/after
     (spglib, 1e-3 A), min interatomic distance > 1.8 A, |P| < 100 bar
     on every component;
  4. equivalence gate: independent single-point re-read of the relaxed
     .lmps must reproduce the minimizer energy to < 1e-8 eV/atom.
All results land in <data_tree>/<thread_dir>/RELAX/<name>/ plus one
summary line per structure in relax-summary_<pot>.yaml.

Usage: python -m hull0k.relax configs/Ni-Cu.yaml <potential-key> [names...]
Optional: --maxiter N (probe runs), names filter.

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.data import atomic_numbers
from ase.io import read as ase_read

from .build_pool import load_config
from .lmps_io import spacegroup_of

REPO = Path(__file__).resolve().parent.parent
MIN_DIST_A = 1.8
PRESS_TOL_BAR = 100.0
EQUIV_TOL_EV_PER_ATOM = 1e-8
FTOL_PER_DOF = 1.0e-8
MAX_FALLBACK_CYCLES = 6

# Which pressure components each box/relax coupling actually relaxes, and
# therefore which ones the pressure gate may test. THE one definition:
# relax.py applies it directly, prepare.py bakes it into job.json for
# run_one.py, and harvest.py reads it back from there. Two copies of this
# mapping -- or a gate that tests the scalar pressure instead of it -- makes
# the integrated and decoupled runners stop at different structures on
# aniso and tri cells, where a box can average to zero pressure while its
# components do not.
PRESS_GATE_COMPONENTS = {
    "iso":   ["press"],
    "aniso": ["pxx", "pyy", "pzz"],
    "tri":   ["press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"],
}


def press_gate_components(coupling):
    """The pressure components the gate tests for this box coupling."""
    if coupling not in PRESS_GATE_COMPONENTS:
        raise SystemExit(f"unknown box coupling '{coupling}'; known: "
                         f"{sorted(PRESS_GATE_COMPONENTS)}")
    return list(PRESS_GATE_COMPONENTS[coupling])


def press_gate(res, coupling, tol=PRESS_TOL_BAR):
    """True when every component this coupling relaxes is below `tol` (bar)."""
    return all(res.get(f"{k}_bar") is not None and abs(res[f"{k}_bar"]) < tol
               for k in press_gate_components(coupling))


def render(skel_path, mapping):
    text = skel_path.read_text()
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    leftovers = re.findall(r"\b(RUNLABEL|THREADDIR|RUNDATE|DATAFILE|PAIRSTYLELINE|"
                           r"PAIRCOEFFLINE|COUPLING|MAXITER|MAXEVAL|BOXPREP)\b", text)
    if leftovers:
        raise RuntimeError(f"unsubstituted placeholders {set(leftovers)} in {skel_path.name}")
    return text


def run_lmp(lmp, infile, cwd, launch=""):
    """launch: optional launcher prefix from the config (`lmp_launch`),
    e.g. "srun -n 4" on a cluster; empty = run the binary directly."""
    cmd = (launch.split() if launch else []) + [lmp, "-in", infile.name, "-screen", "none"]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1200)
    return r.returncode


def parse_results(log_path, label):
    out = {}
    stop = None
    for line in log_path.read_text().splitlines():
        m = re.match(rf"RESULT {re.escape(label)} (\S+) = (\S+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
        if line.startswith("Stopping criterion"):
            stop = line.split("=", 1)[1].strip()
    out["stopping_criterion"] = stop
    return out


def read_lmps(path, elements):
    return ase_read(str(path), format="lammps-data", atom_style="atomic",
                    Z_of_type={i + 1: atomic_numbers[el] for i, el in enumerate(elements)})


def pair_lines(pot):
    style = f"pair_style {pot['pair_style']}"
    coeff = "pair_coeff " + pot["pair_coeff"].replace("POTPATH", pot["file"])
    return style, coeff


def relax_one(cfg, potkey, name, maxiter=None):
    pot = cfg["potentials"][potkey]
    elements = cfg["elements"]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    thread = Path(cfg["data_tree"]) / pot["thread_dir"]
    rundir = thread / "RELAX" / name
    rundir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    with open(pool / f"{name}.yaml") as f:
        side = yaml.safe_load(f)
    coupling = side["box_coupling"]
    natoms = side["natoms"]
    src = pool / f"{name}.lmps"
    shutil.copy2(src, rundir / f"{name}.lmps")

    if maxiter is None:
        maxiter = max(2000, 20 * natoms)
    maxeval = 2 * maxiter
    style_line, coeff_line = pair_lines(pot)
    label = f"{name}_{potkey}"
    boxprep = "change_box all triclinic" if coupling == "tri" else ""

    common = {
        "THREADDIR": pot["thread_dir"], "RUNDATE": today,
        "PAIRSTYLELINE": style_line, "PAIRCOEFFLINE": coeff_line,
        "MAXITER": maxiter, "MAXEVAL": maxeval,
    }

    def do_boxrelax(datafile, tag):
        lab = f"{label}{tag}"
        mapping = dict(common, RUNLABEL=lab, DATAFILE=datafile,
                       COUPLING=coupling, BOXPREP=boxprep)
        infile = rundir / f"{lab}-min.in"
        infile.write_text(render(REPO / "templates/minimize-boxrelax.in.skel", mapping))
        rc = run_lmp(cfg["lmp_binary"], infile, rundir, launch=cfg.get("lmp_launch") or "")
        if rc != 0:
            return rc, {}, None
        res = parse_results(rundir / f"{lab}-min.log", lab)
        return rc, res, f"{lab}-relaxed.lmps"

    def do_fixedcell(datafile, tag):
        lab = f"{label}{tag}"
        mapping = dict(common, RUNLABEL=lab, DATAFILE=datafile)
        infile = rundir / f"{lab}-fixmin.in"
        infile.write_text(render(REPO / "templates/minimize-fixedcell.in.skel", mapping))
        rc = run_lmp(cfg["lmp_binary"], infile, rundir, launch=cfg.get("lmp_launch") or "")
        if rc != 0:
            return rc, {}, None
        res = parse_results(rundir / f"{lab}-fixmin.log", lab)
        return rc, res, f"{lab}-fixmin-relaxed.lmps"

    def converged(res):
        s = res.get("stopping_criterion") or ""
        return ("force tolerance" in s.lower()
                or res.get("fnorm_per_dof_eV_A", 1.0) < FTOL_PER_DOF)

    # the pressure gate tests only the components the coupling relaxes;
    # unrelaxed shear on aniso cells is recorded as a diagnostic ("record
    # the residual anisotropy as a diagnostic")
    relaxed_comps = press_gate_components(coupling)

    # --- stage 1: one-shot box/relax
    path_taken = "boxrelax"
    rc, res, relaxed = do_boxrelax(f"{name}.lmps", "")
    press_ok = rc == 0 and res and press_gate(res, coupling)
    if rc != 0 or not converged(res) or not press_ok:
        # --- two-stage fallback: alternate fixed-cell and box-relax
        path_taken = "twostage-fallback"
        datafile = f"{name}.lmps"
        ok = False
        for cyc in range(MAX_FALLBACK_CYCLES):
            rc1, res1, rel1 = do_fixedcell(datafile, f"-c{cyc}a")
            if rc1 != 0:
                break
            rc, res, relaxed = do_boxrelax(rel1, f"-c{cyc}b")
            if rc != 0:
                break
            datafile = relaxed
            # Exit on the SAME criterion stage 1 used, and the same one
            # run_one.py uses in the decoupled path: the pressure components
            # this coupling actually relaxes. Testing the scalar `press`
            # here instead would diverge for aniso and tri cells, where a
            # cell can average to zero pressure while pxx/pyy/pzz do not --
            # the two execution paths would then take different numbers of
            # fallback cycles and stop at different structures.
            if converged(res) and press_gate(res, coupling):
                ok = True
                break
        if rc != 0 or not res:
            return {"name": name, "potential": potkey, "status": "FAILED",
                    "path": path_taken}
        if not ok:
            path_taken += "-UNCONVERGED"

    # --- integrity checks
    at_in = read_lmps(rundir / f"{name}.lmps", elements)
    at_out = read_lmps(rundir / relaxed, elements)
    sg_before = spacegroup_of(at_in)
    sg_after = spacegroup_of(at_out)
    dmin = float(np.min(at_out.get_all_distances(mic=True)[np.triu_indices(len(at_out), 1)])) \
        if len(at_out) > 1 else np.inf
    pcomps = {k: res.get(f"{k}_bar") for k in ("press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")}
    press_ok = press_gate(res, coupling)
    shear_residual = {k: pcomps.get(k) for k in ("pxy", "pxz", "pyz")
                      if k not in relaxed_comps}

    # --- equivalence gate: independent single point on the relaxed file
    splab = f"{label}-equiv"
    mapping = dict(common, RUNLABEL=splab, DATAFILE=relaxed)
    spfile = rundir / f"{splab}-singlepoint.in"
    spfile.write_text(render(REPO / "templates/singlepoint.in.skel", mapping))
    rc_sp = run_lmp(cfg["lmp_binary"], spfile, rundir, launch=cfg.get("lmp_launch") or "")
    equiv = None
    if rc_sp == 0:
        sp = parse_results(rundir / f"{splab}-singlepoint.log", splab)
        equiv = abs(sp["E_total_eV"] - res["E_total_eV"]) / natoms

    summary = {
        "name": name, "potential": potkey, "status": "OK",
        "path": path_taken,
        "natoms": natoms,
        "E_total_eV": res["E_total_eV"],
        "E_per_atom_eV": res["E_total_eV"] / natoms,
        "fnorm_per_dof_eV_A": res.get("fnorm_per_dof_eV_A"),
        "fnorm_per_atom_eV_A": res.get("fnorm_per_atom_eV_A"),
        "press_components_bar": {k: (None if v is None else round(v, 3)) for k, v in pcomps.items()},
        "press_gate_components": list(relaxed_comps),
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
        "relaxed_file": relaxed,
        "coupling": coupling,
        "date": today,
    }
    with open(rundir / "run.yaml", "w") as f:
        yaml.safe_dump({
            "potential": {"key": potkey, "file": pot["file"], "sha256": pot["sha256"]},
            "lmp_binary": cfg["lmp_binary"],
            "lmp_launch": cfg.get("lmp_launch") or "",
            "lmp_version": cfg.get("lmp_version",
                                   "unrecorded -- see the run log header"),
            "template": "minimize-boxrelax.in.skel v1",
            "result": summary,
        }, f, sort_keys=False)
    return summary


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    maxiter = None
    for a in argv:
        if a.startswith("--maxiter"):
            maxiter = int(a.split("=", 1)[1])
    cfg = load_config(args[0])
    potkey = args[1]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    names = args[2:] or sorted(y.stem for y in pool.glob("*.yaml"))
    thread = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"]
    thread.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        s = relax_one(cfg, potkey, name, maxiter=maxiter)
        summaries.append(s)
        flag = "" if s.get("status") == "OK" else "  <-- " + s.get("status", "?")
        drift = "" if s.get("symmetry_kept", True) else "  [SYMMETRY CHANGED]"
        e = s.get("E_per_atom_eV")
        e_str = "n/a" if e is None else f"{e:.6f}"
        print(f"  {name:34s} {s.get('path','?'):22s} E/at={e_str} "
              f"equiv={s.get('equiv_gate_eV_per_atom')}{flag}{drift}", flush=True)
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

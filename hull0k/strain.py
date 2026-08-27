"""Phase 4: strained (epitaxial) hull -- driver.

Orientation-specific logic (substrate reference, film-cell construction,
prototype eligibility) lives in hull0k/epitaxy.py as a pluggable model
selected via the config (`epitaxy: model: ...`); this driver only walks
the pool, calls the model, renders templates/minimize-epitaxial.in.skel
(fix box/relax z 0.0) and runs LAMMPS. See epitaxy.py for how to add a
new orientation -- this file needs no change for that.

SQS cells are included only with --include-sqs; defect cells only with
--include-defects. Skips are recorded with reasons in
<strain-dir>/skipped-candidates.yaml, where <strain-dir> is named by the
selected epitaxy model (hull0k.epitaxy.strain_dirname) -- so two
orientations never write into each other. Re-runs MERGE into the existing
per-structure summaries, so refinement eps grids extend earlier ones.

Usage: python -m hull0k.strain configs/Ni-Cu.yaml <potential-key> \
           [--eps=-0.04,...] [--include-sqs] [--include-defects] [names...]

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import sys
from pathlib import Path

import yaml

from .build_pool import load_config
from .epitaxy import get_model, strain_dirname
from .lmps_io import write_candidate
from .relax import (REPO, pair_lines, parse_results, read_lmps, render,
                    run_lmp, FTOL_PER_DOF)

DEFAULT_EPS = [x / 100.0 for x in range(-4, 5)]


def strain_one(cfg, potkey, name, eps_list, maxiter=20000):
    pot = cfg["potentials"][potkey]
    elements = cfg["elements"]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    thread = Path(cfg["data_tree"]) / pot["thread_dir"]
    today = datetime.date.today().isoformat()
    model_name, model = get_model(cfg)
    sdir = thread / strain_dirname(cfg, model)
    ctx = model["ref"](cfg, potkey, thread, elements)

    with open(pool / f"{name}.yaml") as f:
        side = yaml.safe_load(f)
    with open(thread / "RELAX" / name / "run.yaml") as f:
        bulk = yaml.safe_load(f)["result"]
    at0 = read_lmps(thread / "RELAX" / name / bulk["relaxed_file"], elements)
    style_line, coeff_line = pair_lines(pot)
    launch = cfg.get("lmp_launch") or ""
    rows = []
    film_rec = {}
    for eps in eps_list:
        tag = f"eps{eps:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
        rundir = sdir / name / tag
        rundir.mkdir(parents=True, exist_ok=True)
        at, film_rec = model["film"](at0, side, eps, ctx)
        at.info.update(prototype=side["prototype"], box_coupling="z-only")
        cname = f"{name}_{tag}"
        write_candidate(at, cname, rundir, elements,
                        source=f"epitaxial film, model {model_name}, "
                               f"eps={eps:+.3f}",
                        extra=dict(film_rec, epitaxy_model=model_name,
                                   model_description=model["description"]))
        label = f"{cname}_{potkey}"
        mapping = {
            "THREADDIR": pot["thread_dir"], "RUNDATE": today,
            "PAIRSTYLELINE": style_line, "PAIRCOEFFLINE": coeff_line,
            "MAXITER": maxiter, "MAXEVAL": 2 * maxiter,
            "RUNLABEL": label, "DATAFILE": f"{cname}.lmps",
        }
        infile = rundir / f"{label}-epimin.in"
        infile.write_text(render(REPO / "templates/minimize-epitaxial.in.skel", mapping))
        rc = run_lmp(cfg["lmp_binary"], infile, rundir, launch=launch)
        if rc != 0:
            rows.append({"eps": eps, "status": "FAILED"})
            continue
        res = parse_results(rundir / f"{label}-epimin.log", label)
        conv = (res.get("stopping_criterion") or "").lower().find("force tolerance") >= 0 \
            or res.get("fnorm_per_dof_eV_A", 1.0) < FTOL_PER_DOF
        rows.append({
            "eps": eps, "status": "OK" if conv else "UNCONVERGED",
            "E_total_eV": res["E_total_eV"],
            "E_per_atom_eV": res["E_per_atom_eV"],
            "pzz_bar": res.get("pzz_bar"),
            "pxx_bar": res.get("pxx_bar"), "pyy_bar": res.get("pyy_bar"),
            "lz_A": res.get("lz_A"),
            "fnorm_per_dof_eV_A": res.get("fnorm_per_dof_eV_A"),
        })

    # merge with any existing summary so refinement runs extend the grid
    sumfile = sdir / name / f"strain-summary_{name}.yaml"
    if sumfile.exists():
        with open(sumfile) as f:
            old = yaml.safe_load(f)
        merged = {round(r["eps"], 6): r for r in old.get("runs", [])}
    else:
        merged = {}
    for r in rows:
        merged[round(r["eps"], 6)] = r
    rows = [merged[k] for k in sorted(merged)]
    rec = {"name": name, "potential": potkey, "date": today,
           "epitaxy_model": model_name,
           "model_description": model["description"],
           "substrate": ctx.get("substrate", "unrecorded"),
           **{k: film_rec[k] for k in ("a0_ref_A", "ncells_inplane", "orientation")
              if k in film_rec},
           "eps_grid": sorted(merged), "runs": rows}
    with open(sumfile, "w") as f:
        yaml.safe_dump(rec, f, sort_keys=False)
    return rec


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    eps_list = DEFAULT_EPS
    for a in argv:
        if a.startswith("--eps="):
            eps_list = [float(v) for v in a.split("=", 1)[1].split(",")]
    include_sqs = "--include-sqs" in argv
    include_defects = "--include-defects" in argv
    cfg = load_config(args[0])
    potkey = args[1]
    _, model = get_model(cfg)
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    thread = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"]
    sdir = thread / strain_dirname(cfg, model)
    if args[2:]:
        names = args[2:]
    else:
        names = []
        skipped = []
        for y in sorted(pool.glob("*.yaml")):
            with open(y) as f:
                side = yaml.safe_load(f)
            proto = side.get("prototype")
            if side.get("note", "").startswith("perfect host reference"):
                skipped.append((y.stem, "bulk finite-size reference, not a hull candidate"))
            elif side.get("defect"):
                if include_defects and proto in model["eligible"]:
                    names.append(y.stem)
                else:
                    skipped.append((y.stem, "defect cell: run only with --include-defects"))
            elif proto in model["eligible_sqs"]:
                if include_sqs:
                    names.append(y.stem)
                else:
                    skipped.append((y.stem, "SQS: supported, run only with --include-sqs"))
            elif str(proto).endswith("_supercell"):
                skipped.append((y.stem, "bulk supercell reference, not a hull candidate"))
            elif proto in model["eligible"]:
                names.append(y.stem)
            else:
                skipped.append((y.stem, f"prototype {proto}: not orientable by the "
                                        "selected epitaxy model (recorded choice)"))
        sdir.mkdir(parents=True, exist_ok=True)
        with open(sdir / "skipped-candidates.yaml", "w") as f:
            yaml.safe_dump({"date": datetime.date.today().isoformat(),
                            "skipped": [{"name": n, "reason": w} for n, w in skipped]},
                           f, sort_keys=False)
        for n, why in skipped:
            print(f"  SKIP {n}: {why}")
    for name in names:
        rec = strain_one(cfg, potkey, name, eps_list)
        ok = sum(1 for r in rec["runs"] if r["status"] == "OK")
        print(f"  {name}: {ok}/{len(rec['runs'])} eps points converged", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Verification tool (brief section 5): re-relax a candidate from perturbed
starts to confirm a hull-touching or otherwise surprising energy.

Each seed: rattle positions (0.05 A gaussian), isotropically scale the
cell by +/-1 %, re-relax with the standard box-relax template, compare
the final energy per atom against the unperturbed relaxation. Results in
<thread_dir>/VERIFY-<name>/verify-<name>.yaml -- deliberately NOT merged
into relax-summary so the hull table stays clean.

Usage: python -m hull0k.verify configs/Ni-Cu.yaml <potential-key> <name> [nseeds]

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

from .build_pool import load_config
from .lmps_io import spacegroup_of, write_candidate
from .relax import REPO, pair_lines, parse_results, read_lmps, render, run_lmp


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    name = argv[2]
    nseeds = int(argv[3]) if len(argv) > 3 else 3
    pot = cfg["potentials"][potkey]
    elements = cfg["elements"]
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    thread = Path(cfg["data_tree"]) / pot["thread_dir"]
    vdir = thread / f"VERIFY-{name}"
    vdir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    with open(pool / f"{name}.yaml") as f:
        side = yaml.safe_load(f)
    with open(thread / "RELAX" / name / "run.yaml") as f:
        ref = yaml.safe_load(f)["result"]

    style_line, coeff_line = pair_lines(pot)
    results = []
    for seed in range(1, nseeds + 1):
        at = read_lmps(pool / f"{name}.lmps", elements)
        rng = np.random.default_rng(seed)
        at.rattle(stdev=0.05, seed=seed)
        scale = 1.0 + (0.01 if seed % 2 else -0.01)
        at.set_cell(at.get_cell() * scale, scale_atoms=True)
        at.info.update(prototype=side["prototype"], box_coupling=side["box_coupling"])
        pname = f"{name}_pert{seed}"
        write_candidate(at, pname, vdir, elements,
                        source=f"perturbed restart of {name} (seed {seed}, scale {scale})")
        label = f"{pname}_{potkey}"
        boxprep = "change_box all triclinic" if side["box_coupling"] == "tri" else ""
        mapping = {
            "THREADDIR": pot["thread_dir"], "RUNDATE": today,
            "PAIRSTYLELINE": style_line, "PAIRCOEFFLINE": coeff_line,
            "MAXITER": 20000, "MAXEVAL": 40000,
            "RUNLABEL": label, "DATAFILE": f"{pname}.lmps",
            "COUPLING": side["box_coupling"], "BOXPREP": boxprep,
        }
        infile = vdir / f"{label}-min.in"
        infile.write_text(render(REPO / "templates/minimize-boxrelax.in.skel", mapping))
        rc = run_lmp(cfg["lmp_binary"], infile, vdir, launch=cfg.get("lmp_launch") or "")
        if rc != 0:
            results.append({"seed": seed, "status": "FAILED"})
            continue
        res = parse_results(vdir / f"{label}-min.log", label)
        at_out = read_lmps(vdir / f"{label}-relaxed.lmps", elements)
        e_at = res["E_total_eV"] / side["natoms"]
        results.append({
            "seed": seed, "status": "OK",
            "E_per_atom_eV": e_at,
            "delta_to_reference_eV_per_atom": float(f"{e_at - ref['E_per_atom_eV']:.3e}"),
            "fnorm_per_dof_eV_A": res.get("fnorm_per_dof_eV_A"),
            "spacegroup_relaxed": spacegroup_of(at_out),
        })
        print(f"  seed {seed}: E/at = {e_at:.8f} eV  (ref {ref['E_per_atom_eV']:.8f}, "
              f"delta {e_at - ref['E_per_atom_eV']:+.2e})  sg = {results[-1]['spacegroup_relaxed']}")

    with open(vdir / f"verify-{name}.yaml", "w") as f:
        yaml.safe_dump({
            "name": name, "potential": potkey, "date": today,
            "reference_E_per_atom_eV": ref["E_per_atom_eV"],
            "reference_spacegroup": ref["spacegroup_after"],
            "perturbation": "rattle 0.05 A + cell scale +/-1 %",
            "seeds": results,
        }, f, sort_keys=False)
    print(f"wrote {vdir / f'verify-{name}.yaml'}")


if __name__ == "__main__":
    main(sys.argv[1:])

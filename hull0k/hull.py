"""Phase 3: hull construction, chemical potentials, defect energies, outputs.

Formation energy per atom: E_f(x) = E(x) - x*E_B - (1-x)*E_A with
x = N_B/(N_A+N_B) counting atoms only; end-member references are THIS
potential's relaxed fcc A and fcc B energies (or the lattices given in the
config key `reference_lattice`, e.g. bcc Nb).

Hull: pymatgen PhaseDiagram, cross-checked against a direct
scipy.spatial.ConvexHull lower hull -- the vertex sets must agree.

Outputs (into <data_tree>/<thread_dir>/):
  hull-table_<pot>_<sys>.dat   (interchange schema, brief section 5)
  hull-plot_<pot>_<sys>.pdf
  mu-vs-x_<pot>_<sys>.dat + mu-vs-x_<pot>_<sys>.pdf
  RESULTS_<pot>.md
  hull-analysis_<pot>.yaml     (machine-readable everything)

Usage: python -m hull0k.hull configs/Ni-Cu.yaml <potential-key>

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

from .build_pool import load_config, ref_lattice
from .labels import is_bulk_candidate, is_sqs
from .relax import read_lmps

EAH_STABLE_TOL = 1e-9  # eV/atom: below this an entry counts as on-hull
# Upper cut of the hull plot's zoom panel (eV/atom). Override per project
# with `plots: {ef_zoom_max: ...}` in the config; the number of clipped
# entries and the true maximum are always stated in the panel title.
EF_ZOOM_MAX_DEFAULT = 0.16


def collect_entries(cfg, potkey):
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    thread = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"]
    with open(thread / f"relax-summary_{potkey}.yaml") as f:
        summaries = {s["name"]: s for s in yaml.safe_load(f)}
    A, B = cfg["elements"]
    entries = []
    for y in sorted(pool.glob("*.yaml")):
        name = y.stem
        if name not in summaries or summaries[name].get("status") != "OK":
            continue
        with open(y) as f:
            side = yaml.safe_load(f)
        s = summaries[name]
        rundir = thread / "RELAX" / name
        at = read_lmps(rundir / s["relaxed_file"], cfg["elements"])
        cellpar = at.cell.cellpar()
        nA = side["composition"][A]
        nB = side["composition"][B]
        mp_ef = side.get("mp_formation_energy_per_atom_eV")
        if mp_ef is None and side.get("mp_matches"):
            mp_ef = side["mp_matches"][0].get("mp_formation_energy_per_atom_eV")
        gates_ok = bool(s.get("equiv_ok") and s.get("press_ok") and s.get("min_dist_ok")
                        and "UNCONVERGED" not in s.get("path", ""))
        entries.append({
            "name": name,
            "prototype": side.get("prototype", "unknown"),
            "source": side.get("source", "unknown").split()[0],
            "nA": nA, "nB": nB, "n": nA + nB,
            "x": nB / (nA + nB),
            "E_per_atom": s["E_per_atom_eV"],
            "E_total": s["E_total_eV"],
            "cellpar": [float(v) for v in cellpar],
            "sg_relaxed": s.get("spacegroup_after", "?"),
            "drifted": 0 if s.get("symmetry_kept", True) else 1,
            "gates_ok": gates_ok,
            "path": s.get("path", "?"),
            "mp_ef": mp_ef,
            "defect": side.get("defect"),
            "host": side.get("host"),
            "solute": side.get("solute"),
        })
    return entries


def formation_energies(entries, cfg):
    A, B = cfg["elements"]
    eA = next(e for e in entries if e["name"] == f"{ref_lattice(cfg, A)}_{A}")
    eB = next(e for e in entries if e["name"] == f"{ref_lattice(cfg, B)}_{B}")
    EA, EB = eA["E_per_atom"], eB["E_per_atom"]
    for e in entries:
        e["Ef"] = e["E_per_atom"] - e["x"] * EB - (1 - e["x"]) * EA
    return EA, EB


def hull_pymatgen(entries, cfg):
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    A, B = cfg["elements"]
    pentries = []
    for e in entries:
        comp = Composition({A: e["nA"], B: e["nB"]})
        pentries.append((e, ComputedEntry(comp, e["E_total"], entry_id=e["name"])))
    pd = PhaseDiagram([pe for _, pe in pentries])
    for e, pe in pentries:
        e["e_above_hull"] = float(pd.get_e_above_hull(pe))
        decomp = pd.get_decomposition(pe.composition)
        e["decomposition"] = {ent.entry_id: round(float(frac), 6)
                              for ent, frac in decomp.items()}
    stable = sorted({ent.entry_id for ent in pd.stable_entries})
    return pd, stable


def hull_scipy(entries):
    """Direct lower convex hull in (x, Ef). Returns the vertex names."""
    from scipy.spatial import ConvexHull

    # per composition keep the lowest-Ef entry (the hull only sees those)
    best = {}
    for e in entries:
        key = round(e["x"], 12)
        if key not in best or e["Ef"] < best[key]["Ef"]:
            best[key] = e
    pts = sorted(((e["x"], e["Ef"], e["name"]) for e in best.values()))
    arr = np.array([(x, y) for x, y, _ in pts])
    if len(arr) < 3:
        return [n for _, _, n in pts]
    hull = ConvexHull(arr)
    verts = set()
    n = len(arr)
    for simplex, eqn in zip(hull.simplices, hull.equations):
        # lower hull facets have outward normal pointing down (eqn[1] < 0)
        if eqn[1] < -1e-12:
            verts.update(simplex)
    names = sorted(pts[i][2] for i in verts)
    return names


def mu_curves(entries, cfg, EA, EB):
    """Chemical potentials from the hull tangent construction.
    On a hull segment with slope s = dEf/dx: mu_B^f = Ef + (1-x)s,
    mu_A^f = Ef - x s (formation-referenced); absolute mu adds the
    end-member energies."""
    by_x = {}
    for e in entries:
        if e["e_above_hull"] < EAH_STABLE_TOL:
            key = round(e["x"], 12)
            if key not in by_x or e["Ef"] < by_x[key][1]:
                by_x[key] = (e["x"], e["Ef"], e["name"])
    hull_pts = sorted(by_x.values())
    segs = []
    for (x1, f1, n1), (x2, f2, n2) in zip(hull_pts[:-1], hull_pts[1:]):
        s = (f2 - f1) / (x2 - x1)
        muB_f = f1 + (1 - x1) * s
        muA_f = f1 - x1 * s
        segs.append({"x_range": [x1, x2], "between": [n1, n2],
                     "slope_eV": s,
                     "mu_A_eV": EA + muA_f, "mu_B_eV": EB + muB_f,
                     "mu_A_f_eV": muA_f, "mu_B_f_eV": muB_f,
                     "dmu_eV": (EB + muB_f) - (EA + muA_f)})
    return hull_pts, segs


def defect_energies(entries, cfg, EA, EB):
    """Vacancy and dilute-substitution energies, reservoirs = the relaxed
    reference end members of THIS potential (fcc unless `reference_lattice`
    says otherwise; constant-mu convention)."""
    A, B = cfg["elements"]
    Ecoh = {A: EA, B: EB}
    out = {}
    for e in entries:
        if not e.get("defect"):
            continue
        # host and solute come from the sidecar the pool builder wrote.
        # The composition is only a fallback for hand-made defect cells:
        # inferring the host from x < 0.5 breaks for a defect in a cell
        # that is not dilute.
        host = e.get("host") or (A if e["x"] < 0.5 else B)
        sol = e.get("solute") or (B if host == A else A)
        n = e["n"]
        if e["defect"] == "vacancy":
            ev = e["E_total"] - n * Ecoh[host]
            out[f"Evac_{host}_{n + 1}sites"] = round(float(ev), 6)
        elif e["defect"] == "substitutional":
            es = e["E_total"] - (n - 1) * Ecoh[host] - Ecoh[sol]
            out[f"Esub_{sol}in{host}_{n}sites"] = round(float(es), 6)
            # identical to n * Ef of the cell: the dilute hull slope
            out[f"dilute_slope_{sol}in{host}_{n}sites"] = round(float(n * e["Ef"]), 6)
    # SQS chord slopes for the dilute-limit consistency check, on whatever
    # parent lattice the SQS cells were generated for
    for e in entries:
        if is_sqs(e["prototype"]):
            x = e["x"]
            if x <= 0.5:
                out[f"sqs_chord_slope_x0_to_{x:.2f}"] = round(float(e["Ef"] / x), 6)
            if x >= 0.5:
                out[f"sqs_chord_slope_x1_to_{x:.2f}"] = round(float(e["Ef"] / (1 - x)), 6)
    return out


def write_outputs(cfg, potkey, entries, stable, scipy_verts, segs, defects, EA, EB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B = cfg["elements"]
    sysname = cfg["system"]
    thread = Path(cfg["data_tree"]) / cfg["potentials"][potkey]["thread_dir"]
    today = datetime.date.today().isoformat()
    pot = cfg["potentials"][potkey]

    # ---- hull table (THE interchange product; schema is a contract)
    tab = thread / f"hull-table_{potkey}_{sysname}.dat"
    with open(tab, "w") as f:
        f.write(f"# hull-table_{potkey}_{sysname}.dat -- 0 K convex-hull table, one row per structure\n")
        f.write(f"# generated {today} by hull0k (convex-hull-0k), potential {potkey} ({pot['file']})\n")
        f.write(f"# sha256(potential) = {pot['sha256']}\n")
        f.write(f"# x = N_{B}/(N_{A}+N_{B}) atom fraction; E and Ef in eV/atom; cell in A and deg\n")
        if any(e["mp_ef"] is not None for e in entries):
            f.write("# Ef_DFT_MP_eV column: DFT formation energies from the Materials Project,\n")
            f.write("#   used for comparison only and NEVER mixed into the hull. That data is\n")
            f.write("#   CC-BY 4.0 -- cite Jain et al., APL Materials 1, 011002 (2013) if you\n")
            f.write("#   publish this column. Per-entry mp_id and mp_url are in the pool sidecars.\n")
        lA, lB = ref_lattice(cfg, A), ref_lattice(cfg, B)
        f.write(f"# Ef(x) = E(x) - x*E_{B},{lB} - (1-x)*E_{A},{lA} with E_{A},{lA} = {EA:.10f}, E_{B},{lB} = {EB:.10f} eV/atom\n")
        f.write("# name prototype source x natoms E_per_atom_eV Ef_per_atom_eV e_above_hull_eV "
                "a_A b_A c_A alpha_deg beta_deg gamma_deg spacegroup_relaxed drifted(0/1) Ef_DFT_MP_eV(or NA)\n")
        for e in sorted(entries, key=lambda q: (q["x"], q["Ef"])):
            cp = e["cellpar"]
            mp = "NA" if e["mp_ef"] is None else f"{e['mp_ef']:.6f}"
            sg = e["sg_relaxed"].replace(" ", "")
            f.write(f"{e['name']} {e['prototype'].replace(' ', '_')} {e['source']} "
                    f"{e['x']:.6f} {e['n']} {e['E_per_atom']:.8f} {e['Ef']:.8f} "
                    f"{e['e_above_hull']:.8f} "
                    f"{cp[0]:.6f} {cp[1]:.6f} {cp[2]:.6f} {cp[3]:.4f} {cp[4]:.4f} {cp[5]:.4f} "
                    f"{sg} {e['drifted']} {mp}\n")

    # ---- mu table
    mut = thread / f"mu-vs-x_{potkey}_{sysname}.dat"
    with open(mut, "w") as f:
        f.write(f"# mu-vs-x_{potkey}_{sysname}.dat -- chemical potentials along the 0 K hull (tangent construction)\n")
        f.write(f"# generated {today} by hull0k, potential {potkey}\n")
        f.write(f"# mu_{A} and mu_{B} absolute (eV/atom, same zero as the potential); dmu = mu_{B} - mu_{A}\n")
        f.write(f"# x_lo x_hi mu_{A}_eV mu_{B}_eV dmu_eV segment_between\n")
        for s in segs:
            f.write(f"{s['x_range'][0]:.6f} {s['x_range'][1]:.6f} "
                    f"{s['mu_A_eV']:.8f} {s['mu_B_eV']:.8f} {s['dmu_eV']:.8f} "
                    f"{s['between'][0]}--{s['between'][1]}\n")

    # ---- hull plot: full range + zoom on the low-Ef regime (named cut,
    # canon rule "give the region of interest its own axis; clip loudly")
    EF_ZOOM_MAX = float((cfg.get("plots") or {}).get("ef_zoom_max",
                                                     EF_ZOOM_MAX_DEFAULT))
    conv = [e for e in entries if e["gates_ok"] and not e["drifted"]]
    prob = [e for e in entries if not (e["gates_ok"] and not e["drifted"])]
    hull_seq = sorted([(e["x"], e["Ef"], e["name"]) for e in entries
                       if e["e_above_hull"] < EAH_STABLE_TOL])
    hull_x = [x for x, _, _ in hull_seq]
    hull_y = [y for _, y, _ in hull_seq]
    mp = [e for e in entries if e["mp_ef"] is not None]
    ef_max = max(e["Ef"] for e in entries)
    n_cut = sum(1 for e in entries if e["Ef"] > EF_ZOOM_MAX)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 8.6), height_ratios=[1, 1.4])
    for ax, zoom in ((ax0, False), (ax1, True)):
        ax.plot(hull_x, hull_y, "-", color="#1f4e8c", lw=2, zorder=2, label="convex hull")
        ax.plot([e["x"] for e in conv], [e["Ef"] for e in conv], "o", ms=6,
                mfc="#1f4e8c", mec="#1f4e8c", ls="none", zorder=3,
                label="candidates (converged)")
        if prob:
            ax.plot([e["x"] for e in prob], [e["Ef"] for e in prob], "o", ms=6,
                    mfc="none", mec="#1f4e8c", ls="none", zorder=3,
                    label="candidates (not assured)")
        if mp:
            ax.plot([e["x"] for e in mp], [e["mp_ef"] for e in mp], "s", ms=5,
                    mfc="none", mec="#999999", ls="none", zorder=1,
                    label="DFT (Materials Project)")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_ylabel("formation energy E_f (eV/atom)")
        if zoom:
            ax.set_ylim(min(min(hull_y), 0) - 0.012, EF_ZOOM_MAX)
            ax.set_xlabel(f"x = N_{B} / (N_{A} + N_{B})")
            ax.set_title(f"zoom: E_f < {EF_ZOOM_MAX} eV/atom "
                         f"({n_cut} entries above cut, max E_f = {ef_max:.3f} eV/atom)",
                         fontsize=9)
            for e in entries:
                near_hull = (e["Ef"] < EF_ZOOM_MAX and e["e_above_hull"] < 0.015
                             and is_bulk_candidate(e["prototype"])
                             and 0 < e["x"] < 1)
                if (e["e_above_hull"] < EAH_STABLE_TOL and 0 < e["x"] < 1) or near_hull:
                    ax.annotate(e["name"], (e["x"], e["Ef"]), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color="#444444")
        else:
            ax.set_title(f"0 K formation-energy hull, {sysname}, {potkey}\n"
                         f"references: relaxed {ref_lattice(cfg, A)} {A} ({EA:.5f}) and {ref_lattice(cfg, B)} {B} ({EB:.5f}) eV/atom",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            for e in entries:
                if e["Ef"] > EF_ZOOM_MAX:
                    ax.annotate(e["name"], (e["x"], e["Ef"]), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color="#444444")
    fig.tight_layout()
    fig.savefig(thread / f"hull-plot_{potkey}_{sysname}.pdf")
    plt.close(fig)

    # ---- mu plot (step curves across segments)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, color, lab in (("mu_A_eV", "#1f4e8c", f"mu_{A}"),
                            ("mu_B_eV", "#b0532a", f"mu_{B}")):
        xs, ys = [], []
        for s in segs:
            xs += s["x_range"]
            ys += [s[key], s[key]]
        ax.plot(xs, ys, "-", lw=2, color=color, label=f"{lab} (eV/atom)")
    ax.set_xlabel(f"x = N_{B} / (N_{A} + N_{B})")
    ax.set_ylabel("chemical potential (eV/atom)")
    ax.set_title(f"Hull chemical potentials (tangent construction), {sysname}, {potkey}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(thread / f"mu-vs-x_{potkey}_{sysname}.pdf")
    plt.close(fig)

    # ---- machine-readable analysis record
    record = {
        "potential": potkey, "date": today,
        "references_eV_per_atom": {A: float(EA), B: float(EB)},
        "hull_members_pymatgen": stable,
        "hull_members_scipy": scipy_verts,
        "hull_crosscheck_agree": sorted(stable) == sorted(scipy_verts),
        "segments": segs,
        "defect_energies_eV": defects,
        "n_entries": len(entries),
    }
    with open(thread / f"hull-analysis_{potkey}.yaml", "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)
    return record


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    entries = collect_entries(cfg, potkey)
    EA, EB = formation_energies(entries, cfg)
    pd, stable = hull_pymatgen(entries, cfg)
    scipy_verts = hull_scipy(entries)
    hull_pts, segs = mu_curves(entries, cfg, EA, EB)
    defects = defect_energies(entries, cfg, EA, EB)
    rec = write_outputs(cfg, potkey, entries, stable, scipy_verts, segs, defects, EA, EB)
    print(f"hull members (pymatgen): {stable}")
    print(f"hull members (scipy):    {scipy_verts}")
    print(f"cross-check agree: {rec['hull_crosscheck_agree']}")
    for k, v in defects.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main(sys.argv[1:])

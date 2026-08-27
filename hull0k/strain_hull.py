"""Phase 4 analysis: strained (epitaxial) hulls from the strain.py sweeps.

Reference conventions (brief section 6), both computed per entry:
  (i)  Ef_unstrained_ref = E(eps) - x*E_B,bulk - (1-x)*E_A,bulk
       (contains ALL elastic energy, end members included);
  (ii) Ef_coherent_ref   = E(eps) - x*E_B(eps) - (1-x)*E_A(eps)
       (the coherent hull: what is stable ON this substrate).
The two differ by a function LINEAR in x, so the lower-hull membership
and e_above_hull are identical for both -- asserted, and one
e_above_hull column is written.

Outputs (in <thread_dir>/<strain-dir>/, named by the selected epitaxy
model -- see hull0k.epitaxy.strain_dirname; nothing about the substrate,
the orientation or the elements is hard-coded here):
  hull-table-strained_<pot>_<sys>_<epstag>.dat   one per eps
  strain-trajectories_<pot>_<sys>.dat            long format, all eps
  strain-stability_<pot>_<sys>.pdf               trajectories + (x,eps) map
  RESULTS-STRAIN_<pot>.md                        prose incl. crossings

Usage: python -m hull0k.strain_hull configs/Ni-Cu.yaml <potential-key>

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

from .build_pool import load_config
from .epitaxy import find_strain_dir, get_model

EAH_TOL = 1e-9  # eV/atom; below this an entry counts as on-hull


def lower_hull_eah(points):
    """points: list of (x, Ef). Returns e_above_hull list via the lower
    convex hull of the per-composition minima."""
    best = {}
    for i, (x, f) in enumerate(points):
        key = round(x, 12)
        if key not in best or f < best[key][1]:
            best[key] = (x, f)
    pts = sorted(best.values())
    # Andrew monotone chain, lower hull
    hull = []
    for p in pts:
        while len(hull) >= 2 and \
                (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= \
                (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    out = []
    for x, f in points:
        y = np.interp(x, hx, hy)
        out.append(f - y)
    return out, hull


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    A, B = cfg["elements"]
    sysname = cfg["system"]
    pot = cfg["potentials"][potkey]
    thread = Path(cfg["data_tree"]) / pot["thread_dir"]
    model_name, model = get_model(cfg)
    sdir = find_strain_dir(cfg, thread, model)
    pool = Path(cfg["data_tree"]) / cfg["pool_dir"]
    today = datetime.date.today().isoformat()

    # bulk (unstrained) end-member references from phase 2
    with open(thread / f"relax-summary_{potkey}.yaml") as f:
        bulk = {s["name"]: s for s in yaml.safe_load(f)}
    EA_bulk = bulk[f"fcc_{A}"]["E_per_atom_eV"]
    EB_bulk = bulk[f"fcc_{B}"]["E_per_atom_eV"]

    # load all strain summaries
    recs = {}
    for sy in sorted(sdir.glob("*/strain-summary_*.yaml")):
        with open(sy) as f:
            rec = yaml.safe_load(f)
        with open(pool / f"{rec['name']}.yaml") as f:
            side = yaml.safe_load(f)
        rec["x"] = side[f"x_{B}"]
        rec["natoms"] = side["natoms"]
        rec["prototype"] = side["prototype"]
        rec["defect"] = side.get("defect")
        rec["host"] = side.get("host")
        rec["solute"] = side.get("solute")
        recs[rec["name"]] = rec
    if not recs:
        raise SystemExit(f"no strain summaries under {sdir}")
    eps_grid = sorted({r["eps"] for rec in recs.values() for r in rec["runs"]})

    # Everything orientation-specific in the tables, plots and prose below
    # comes from the epitaxy model and from what the driver recorded in the
    # summaries -- never from a literal. Ask the model first (authoritative),
    # fall back to the recorded values of an existing sweep.
    try:
        ctx = model["ref"](cfg, potkey, thread, cfg["elements"])
    except Exception:
        ctx = {}

    def _recorded(key):
        for r in recs.values():
            if r.get(key) not in (None, "", "unrecorded"):
                return r[key]
        return None

    substrate = ctx.get("substrate") or _recorded("substrate") or "unrecorded substrate"
    orientation = _recorded("orientation") or "as defined by the epitaxy model"
    model_desc = _recorded("model_description") or model["description"]
    a0_ref = ctx.get("a0_ref") or _recorded("a0_ref_A")
    a0_ref_str = 'unrecorded' if a0_ref is None else f'{float(a0_ref):.6f} A'
    constraint = (f"biaxial epitaxy on {substrate}; in-plane cell vectors FIXED "
                  f"at (1+eps) x the substrate reference (a0_ref = {a0_ref_str}, "
                  f"this potential's own relaxed value); the out-of-plane length "
                  f"and all internal coordinates relax (fix box/relax z 0.0)")

    def e_at(name, eps):
        for r in recs[name]["runs"]:
            if abs(r["eps"] - eps) < 1e-12 and r["status"] == "OK":
                return r["E_per_atom_eV"]
        return None

    # per-eps tables + membership
    members = {}   # eps -> set of hull member names
    rows_long = []
    for eps in eps_grid:
        EA_eps = e_at(f"fcc_{A}", eps)
        EB_eps = e_at(f"fcc_{B}", eps)
        entries = []
        for name, rec in recs.items():
            E = e_at(name, eps)
            if E is None:
                continue
            x = rec["x"]
            ef_i = E - x * EB_bulk - (1 - x) * EA_bulk
            ef_ii = (None if EA_eps is None or EB_eps is None
                     else E - x * EB_eps - (1 - x) * EA_eps)
            entries.append(dict(name=name, prototype=rec["prototype"], x=x,
                                natoms=rec["natoms"], E=E, ef_i=ef_i, ef_ii=ef_ii))
        eah_i, _ = lower_hull_eah([(e["x"], e["ef_i"]) for e in entries])
        eah_ii, _ = lower_hull_eah([(e["x"], e["ef_ii"]) for e in entries])
        assert max(abs(a - b) for a, b in zip(eah_i, eah_ii)) < 1e-9, \
            "e_above_hull differs between reference conventions -- bug"
        for e, eah in zip(entries, eah_i):
            e["eah"] = eah
        members[eps] = {e["name"] for e in entries if e["eah"] < EAH_TOL}

        tag = f"eps{eps:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
        tab = sdir / f"hull-table-strained_{potkey}_{sysname}_{tag}.dat"
        with open(tab, "w") as f:
            f.write(f"# {tab.name} -- strained epitaxial hull table at eps = {eps:+.3f}\n")
            f.write(f"# generated {today} by hull0k, potential {potkey}; epitaxy model "
                    f"{model_name}: {model_desc}\n")
            f.write(f"# substrate {substrate}, a0_ref = {a0_ref_str} (this potential's own "
                    f"relaxed value); orientation {orientation}\n")
            f.write(f"# Ef_unstrained_ref vs bulk end members (E_{A} = {EA_bulk:.10f}, "
                    f"E_{B} = {EB_bulk:.10f} eV/atom)\n")
            f.write(f"# Ef_coherent_ref vs the end members strained the same way "
                    f"(E_{A}(eps) = {EA_eps:.10f}, E_{B}(eps) = {EB_eps:.10f} eV/atom)\n")
            f.write("# e_above_hull is identical for both conventions (they differ by a "
                    "linear-in-x function); one column written\n")
            f.write("# name prototype x natoms eps E_per_atom_eV Ef_unstrained_ref_eV "
                    "Ef_coherent_ref_eV e_above_hull_eV\n")
            for e in sorted(entries, key=lambda q: (q["x"], q["ef_ii"])):
                f.write(f"{e['name']} {e['prototype'].replace(' ', '_')} {e['x']:.6f} "
                        f"{e['natoms']} {eps:+.3f} {e['E']:.8f} {e['ef_i']:.8f} "
                        f"{e['ef_ii']:.8f} {e['eah']:.8f}\n")
        for e in entries:
            rows_long.append((e["name"], eps, e["x"], e["ef_i"], e["ef_ii"], e["eah"],
                              1 if e["eah"] < EAH_TOL else 0))

    # long-format trajectories
    traj = sdir / f"strain-trajectories_{potkey}_{sysname}.dat"
    with open(traj, "w") as f:
        f.write(f"# {traj.name} -- e_above_hull and formation-energy trajectories vs eps\n")
        f.write(f"# generated {today} by hull0k, potential {potkey}; epitaxy model "
                f"{model_name}, substrate {substrate}, orientation {orientation}\n")
        f.write("# name eps x Ef_unstrained_ref_eV Ef_coherent_ref_eV e_above_hull_eV hull_member(0/1)\n")
        for r in rows_long:
            f.write(f"{r[0]} {r[1]:+.3f} {r[2]:.6f} {r[3]:.8f} {r[4]:.8f} {r[5]:.8f} {r[6]}\n")

    # strained defect energies: reservoirs = the fcc end members strained
    # the SAME way at the same eps (coherent convention, stated in the file)
    defect_rows = []
    defect_keys = set()
    for eps in eps_grid:
        EA_eps = e_at(f"fcc_{A}", eps)
        EB_eps = e_at(f"fcc_{B}", eps)
        row = {"eps": eps}
        for name, rec in recs.items():
            if not rec.get("defect"):
                continue
            e_pa = e_at(name, eps)
            if e_pa is None or EA_eps is None or EB_eps is None:
                continue
            n = rec["natoms"]
            E_cell = e_pa * n
            # host/solute come from the sidecar the pool builder wrote; the
            # composition is only a fallback for hand-made defect cells.
            host = rec.get("host") or (A if rec["x"] < 0.5 else B)
            sol = rec.get("solute") or (B if host == A else A)
            Ehost = EA_eps if host == A else EB_eps
            Esol = EA_eps if sol == A else EB_eps
            if rec["defect"] == "vacancy":
                key = f"Evac_{host}_{n + 1}sites_eV"
                row[key] = E_cell - n * Ehost
            else:
                key = f"Esub_{sol}in{host}_{n}sites_eV"
                row[key] = E_cell - (n - 1) * Ehost - Esol
            defect_keys.add(key)
        defect_rows.append(row)
    defect_keys = sorted(defect_keys)
    if defect_keys:
        dft = sdir / f"defect-energies-vs-eps_{potkey}_{sysname}.dat"
        with open(dft, "w") as f:
            f.write(f"# {dft.name} -- defect energies under epitaxy on {substrate} "
                    f"({orientation})\n")
            f.write(f"# generated {today} by hull0k, potential {potkey}\n")
            f.write("# reservoir convention: fcc end members strained the SAME way at the\n")
            f.write("# same eps (coherent references); energies in eV per defect\n")
            f.write("# eps " + " ".join(defect_keys) + "\n")
            for row in defect_rows:
                vals = " ".join(f"{row[k]:.6f}" if k in row else "NA" for k in defect_keys)
                f.write(f"{row['eps']:+.3f} {vals}\n")

    # crossings: membership changes between consecutive eps
    crossings = []
    for e0, e1 in zip(eps_grid[:-1], eps_grid[1:]):
        gained = members[e1] - members[e0]
        lost = members[e0] - members[e1]
        for n in sorted(gained):
            crossings.append((e0, e1, n, "enters the hull"))
        for n in sorted(lost):
            crossings.append((e0, e1, n, "leaves the hull"))

    # ---- plot: trajectories + stability map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#1f4e8c", "#b0532a", "#3a7d44", "#7b4b94", "#c28e0e",
               "#3d8b8b", "#8c2d2d", "#5b5b5b", "#94651e", "#4863c7"]
    names_sorted = sorted(recs, key=lambda n: (recs[n]["x"], n))
    color = {n: palette[i % len(palette)] for i, n in enumerate(names_sorted)}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 8.8), height_ratios=[1.3, 1])
    for n in names_sorted:
        xs = [r[1] for r in rows_long if r[0] == n]
        ys = [r[5] * 1000 for r in rows_long if r[0] == n]
        ax0.plot(xs, ys, "-o", ms=4, lw=1.6, color=color[n])
        ax0.annotate(n, (xs[-1], ys[-1]), textcoords="offset points", xytext=(5, 0),
                     fontsize=7, color=color[n])
    ax0.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax0.set_xlabel(f"in-plane strain eps (on {substrate}; {orientation})")
    ax0.set_ylabel("e_above_hull (meV/atom)")
    ax0.set_title(f"Strained-hull trajectories, {sysname}, {potkey}\n"
                  f"epitaxy on {substrate}, in-plane cell fixed at "
                  f"(1+eps)*a0_ref; hull membership identical for both "
                  f"reference conventions", fontsize=9)
    ax0.set_xlim(min(eps_grid) - 0.002, max(eps_grid) + 0.014)

    for n in names_sorted:
        x = recs[n]["x"]
        for eps in eps_grid:
            if any(abs(r["eps"] - eps) < 1e-12 and r["status"] == "OK"
                   for r in recs[n]["runs"]):
                on = n in members[eps]
                ax1.plot(x, eps, "o", ms=9 if on else 4,
                         mfc=color[n] if on else "none", mec=color[n],
                         mew=1.2, ls="none")
    ax1.set_xlabel(f"x = N_{B} / (N_{A} + N_{B})")
    ax1.set_ylabel("eps")
    ax1.set_title("stability map: filled = on the (coherent) hull, open = above it", fontsize=9)
    fig.tight_layout()
    fig.savefig(sdir / f"strain-stability_{potkey}_{sysname}.pdf")
    plt.close(fig)

    # ---- prose
    L = [f"# RESULTS-STRAIN_{potkey}.md -- strained epitaxial hull, {sysname}",
         "",
         f"Project: convex-hull-0k  Thread: {pot['thread_dir']}/{sdir.name}",
         f"Author: Erik Bitzek (created with LLM-LMPS)  Date: {today}",
         "",
         f"Epitaxy model: `{model_name}` -- {model_desc}",
         "",
         f"Constraint: {constraint}. Orientation: {orientation} for all "
         "included candidates -- a recorded CHOICE, not an optimum.",
         f"eps grid: {', '.join(f'{e:+.3f}' for e in eps_grid)}.",
         "Candidate set actually run: "
         + ", ".join(f"`{n}`" for n in sorted(recs)) + ". "
         "Skipped candidates and reasons in `skipped-candidates.yaml`.",
         "",
         "Both reference conventions are in every table: Ef_unstrained_ref "
         "(vs bulk end members; contains all elastic energy) and "
         "Ef_coherent_ref (vs end members strained the same way -- the "
         "coherent hull). They differ by a linear-in-x function, so hull "
         "membership and e_above_hull are identical for both (asserted in "
         "the analysis).",
         "",
         "## Hull membership vs eps",
         "",
         "| eps | hull members |"]
    L.append("|---|---|")
    for eps in eps_grid:
        L.append(f"| {eps:+.3f} | " + ", ".join(f"`{n}`" for n in sorted(members[eps])) + " |")
    L.append("")
    L.append("## Crossings (hull membership changes)")
    L.append("")
    if crossings:
        L.append("**FINDING: strain changes hull membership.**")
        for e0, e1, n, what in crossings:
            L.append(f"- between eps = {e0:+.3f} and {e1:+.3f}: `{n}` {what}.")
    else:
        L.append("No structure enters or leaves the hull anywhere on the eps grid.")
    L.append("")
    if defect_keys:
        L.append("## Defect energies under strain")
        L.append("")
        L.append("Reservoirs: fcc end members strained the SAME way at the same eps "
                 "(coherent convention). Full table: `defect-energies-vs-eps_*.dat`.")
        L.append("")
        L.append("| eps | " + " | ".join(k.replace("_eV", "") for k in defect_keys) + " |")
        L.append("|---|" + "---|" * len(defect_keys))
        for row in defect_rows:
            L.append(f"| {row['eps']:+.3f} | "
                     + " | ".join(f"{row[k]:+.4f}" if k in row else "NA" for k in defect_keys)
                     + " |")
        L.append("")
    L.append("See `strain-trajectories_*.dat`, per-eps `hull-table-strained_*.dat`, "
             "and `strain-stability_*.pdf`.")
    (sdir / f"RESULTS-STRAIN_{potkey}.md").write_text("\n".join(L) + "\n")

    print(f"eps grid: {[f'{e:+.3f}' for e in eps_grid]}")
    for eps in eps_grid:
        print(f"  eps {eps:+.3f}: hull = {sorted(members[eps])}")
    if crossings:
        for c in crossings:
            print(f"  CROSSING between {c[0]:+.3f} and {c[1]:+.3f}: {c[2]} {c[3]}")
    print(f"wrote {sdir / f'RESULTS-STRAIN_{potkey}.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])

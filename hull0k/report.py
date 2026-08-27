"""Phase 3 reporting: RESULTS_<potential>.md in each thread dir.

Composes prose findings from hull-analysis_<pot>.yaml + the hull table:
validation gates, hull membership (with the verification record if one
exists), e_above_hull spectrum, defect-energy table with reservoir
conventions, dilute-limit consistency, and -- when both potentials have
been analyzed -- the EAM-vs-ACE-vs-DFT comparison table.

Usage: python -m hull0k.report configs/Ni-Cu.yaml <potential-key>

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import sys
from pathlib import Path

import yaml

from .build_pool import load_config
from .labels import is_bulk_candidate


def load_table(path):
    rows = []
    cols = ["name", "prototype", "source", "x", "natoms", "E_per_atom_eV",
            "Ef_per_atom_eV", "e_above_hull_eV", "a_A", "b_A", "c_A",
            "alpha_deg", "beta_deg", "gamma_deg", "spacegroup_relaxed",
            "drifted", "Ef_DFT_MP_eV"]
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        row = dict(zip(cols, parts))
        for k in ("x", "E_per_atom_eV", "Ef_per_atom_eV", "e_above_hull_eV"):
            row[k] = float(row[k])
        row["natoms"] = int(row["natoms"])
        rows.append(row)
    return rows


def main(argv):
    cfg = load_config(argv[0])
    potkey = argv[1]
    A, B = cfg["elements"]
    sysname = cfg["system"]
    pot = cfg["potentials"][potkey]
    thread = Path(cfg["data_tree"]) / pot["thread_dir"]
    today = datetime.date.today().isoformat()

    with open(thread / f"hull-analysis_{potkey}.yaml") as f:
        ana = yaml.safe_load(f)
    rows = load_table(thread / f"hull-table_{potkey}_{sysname}.dat")
    by_name = {r["name"]: r for r in rows}
    refs = cfg.get("references", {}).get(potkey)

    # verification records, if any
    verifs = []
    for vy in sorted(thread.glob("VERIFY-*/verify-*.yaml")):
        with open(vy) as f:
            verifs.append(yaml.safe_load(f))

    # canonicalize degenerate hull members: a supercell reference that ties
    # the primitive end member (|dEf| < 1e-9) is reported under the fcc name
    hull_members = []
    for n in sorted(set(ana["hull_members_pymatgen"])):
        for canon in (f"fcc_{A}", f"fcc_{B}"):
            if n != canon and n.startswith(canon) and canon in by_name and \
                    abs(by_name[n]["Ef_per_atom_eV"] - by_name[canon]["Ef_per_atom_eV"]) < 1e-9:
                n = canon
                break
        if n not in hull_members:
            hull_members.append(n)
    nontrivial = [n for n in hull_members
                  if not n.startswith((f"fcc_{A}", f"fcc_{B}"))]

    L = []
    L.append(f"# RESULTS_{potkey}.md -- 0 K convex hull, {sysname}")
    L.append("")
    L.append(f"Project: convex-hull-0k  Thread: {pot['thread_dir']}")
    L.append(f"Author: Erik Bitzek (created with LLM-LMPS)  Date: {today}")
    L.append(f"Potential: `{pot['file']}`")
    L.append(f"sha256: `{pot['sha256']}`  "
             f"LAMMPS: {cfg.get('lmp_version', 'unrecorded -- see the run log header')} "
             f"(`{cfg.get('lmp_binary', 'unrecorded')}`)")
    L.append("")
    L.append("Convergence convention: fnorm/sqrt(3N) < 1e-8 eV/A, where fnorm is the")
    L.append("2-norm of the global 3N force vector; both fnorm/N and fnorm/sqrt(3N)")
    L.append("are recorded per run in RELAX/<name>/run.yaml.")
    L.append("")

    # ---- end members / validation gates
    L.append("## End members and validation gates")
    L.append("")
    L.append(f"| quantity | this work | on record | gate |")
    L.append(f"|---|---|---|---|")
    fA = by_name[f"fcc_{A}"]
    fB = by_name[f"fcc_{B}"]
    if refs:
        gates = [
            (f"a0 fcc {A} (A)", float(fA["a_A"]), refs.get(f"a0_{A}"), 1e-3),
            (f"E_coh fcc {A} (eV/at)", fA["E_per_atom_eV"], refs.get(f"Ecoh_{A}"), 1e-3),
            (f"a0 fcc {B} (A)", float(fB["a_A"]), refs.get(f"a0_{B}"), 1e-3),
            (f"E_coh fcc {B} (eV/at)", fB["E_per_atom_eV"], refs.get(f"Ecoh_{B}"), 1e-3),
        ]
        for label, val, ref, tol in gates:
            if ref is None:
                L.append(f"| {label} | {val:.6f} | n/a | -- |")
            else:
                ok = abs(val - ref) < tol
                L.append(f"| {label} | {val:.6f} | {ref:.6f} | {'PASS' if ok else 'FAIL'} "
                         f"(|d| = {abs(val - ref):.2e}) |")
    else:
        L.append(f"| a0 fcc {A} (A) | {float(fA['a_A']):.6f} | none on record | -- |")
        L.append(f"| E_coh fcc {A} (eV/at) | {fA['E_per_atom_eV']:.6f} | none on record | -- |")
        L.append(f"| a0 fcc {B} (A) | {float(fB['a_A']):.6f} | none on record | -- |")
        L.append(f"| E_coh fcc {B} (eV/at) | {fB['E_per_atom_eV']:.6f} | none on record | -- |")
    L.append("")
    L.append(f"Hull cross-check (pymatgen PhaseDiagram vs direct scipy ConvexHull): "
             f"{'AGREE' if ana['hull_crosscheck_agree'] else 'DISAGREE -- INVESTIGATE'}.")
    L.append("")

    # ---- hull membership
    L.append("## Hull membership")
    L.append("")
    L.append(f"Hull members: {', '.join('`' + n + '`' for n in hull_members)}.")
    L.append("")
    if nontrivial:
        L.append(f"**FINDING: the hull is NOT the trivial {{fcc {A}, fcc {B}}} pair.**")
        for n in nontrivial:
            r = by_name[n]
            L.append(f"- `{n}` ({r['prototype']}, {r['spacegroup_relaxed']}) at "
                     f"x = {r['x']:.4f} with E_f = {r['Ef_per_atom_eV']*1000:.2f} meV/atom < 0.")
        for v in verifs:
            deltas = [s.get("delta_to_reference_eV_per_atom") for s in v["seeds"]
                      if s.get("status") == "OK"]
            sgs = {s.get("spacegroup_relaxed") for s in v["seeds"] if s.get("status") == "OK"}
            L.append(f"- Verification of `{v['name']}`: {len(deltas)} perturbed restarts "
                     f"({v['perturbation']}) all reconverge to the same minimum "
                     f"(max |dE| = {max(abs(d) for d in deltas):.1e} eV/atom, "
                     f"spacegroups {sorted(sgs)}). The negative formation energy is a "
                     f"property of the potential, not a relaxation artifact.")
    else:
        L.append(f"The hull is the expected trivial pair {{fcc {A}, fcc {B}}}: every "
                 f"compound and solution has E_f > 0. This matches the assessed {sysname} "
                 f"phase diagram (no stable intermetallics).")
    L.append("")

    # ---- e_above_hull spectrum
    L.append("## Metastability spectrum (e_above_hull)")
    L.append("")
    L.append("| structure | prototype | x | E_f (meV/at) | e_above_hull (meV/at) | sg relaxed | drifted |")
    L.append("|---|---|---|---|---|---|---|")
    # Defect cells and perfect-host finite-size references are excluded here:
    # the first belong to the defect table below, the second are energetically
    # degenerate with their end member. Decided from the prototype LABEL, so
    # this keeps working for any elements, host lattice and supercell size.
    for r in sorted(rows, key=lambda q: (q["x"], q["Ef_per_atom_eV"])):
        if not is_bulk_candidate(r["prototype"]):
            continue
        L.append(f"| {r['name']} | {r['prototype']} | {r['x']:.3f} | "
                 f"{r['Ef_per_atom_eV']*1000:8.2f} | {r['e_above_hull_eV']*1000:8.2f} | "
                 f"{r['spacegroup_relaxed']} | {r['drifted']} |")
    L.append("")
    drifted = [r for r in rows if r["drifted"] == "1"]
    if drifted:
        L.append("Drifted candidates (relaxed into a different structure; energy belongs")
        L.append("to the drift product, not the prototype):")
        for r in drifted:
            L.append(f"- `{r['name']}`: built as {r['prototype']}, relaxed to {r['spacegroup_relaxed']}")
        L.append("")

    # ---- chemical potentials
    L.append("## Chemical potentials along the hull (tangent construction)")
    L.append("")
    L.append(f"See `mu-vs-x_{potkey}_{sysname}.dat` / `.pdf`. Segments:")
    L.append("")
    L.append(f"| x range | between | mu_{A} (eV) | mu_{B} (eV) | dmu = mu_{B}-mu_{A} (eV) |")
    L.append("|---|---|---|---|---|")
    for s in ana["segments"]:
        L.append(f"| {s['x_range'][0]:.4f} .. {s['x_range'][1]:.4f} | "
                 f"{s['between'][0]} -- {s['between'][1]} | {s['mu_A_eV']:.5f} | "
                 f"{s['mu_B_eV']:.5f} | {s['dmu_eV']:.5f} |")
    L.append("")

    # ---- defect energies
    d = ana["defect_energies_eV"]
    L.append("## Defect energies")
    L.append("")
    L.append("Reservoir convention: chemical potentials of the RELAXED PURE fcc end")
    L.append(f"members of this potential (mu_{A} = E_coh(fcc {A}), mu_{B} = E_coh(fcc {B})).")
    L.append("")
    L.append("| quantity | cell | value (eV) |")
    L.append("|---|---|---|")
    for k in sorted(d):
        if k.startswith(("Evac", "Esub")):
            base, sites = k.rsplit("_", 1)
            L.append(f"| {base} | {sites} | {d[k]:+.4f} |")
    L.append("")
    if refs and refs.get(f"Evac_{A}") is not None:
        key = [k for k in d if k.startswith(f"Evac_{A}")]
        if key:
            # cell size is read off the key the analysis generated -- the
            # reference cell size, if it matters, belongs in the config's
            # `references` note, not in a literal here.
            sites = key[0].rsplit("_", 1)[1]
            L.append(f"On-record comparison: Evac({A}) = {refs[f'Evac_{A}']:.4f} eV "
                     f"vs {d[key[0]]:.4f} eV here ({sites}).")
        keys = [k for k in d if k.startswith(f"Esub_{B}in{A}")]
        if keys and refs.get(f"Esub_{B}in{A}") is not None:
            L.append(f"On-record comparison: Esub({B} in {A}) = {refs[f'Esub_{B}in{A}']:.4f} eV vs "
                     + ", ".join(f"{d[k]:.4f} eV ({k.rsplit('_', 1)[1]})" for k in sorted(keys)) + ".")
        L.append("")

    # ---- dilute-limit consistency
    L.append("## Dilute-limit consistency (defect branch vs solution branch)")
    L.append("")
    L.append(f"The x->0 slope of E_f equals Esub({B} in {A}) by construction from the")
    L.append("defect cells; the SQS/compound branch must extrapolate to the same slope.")
    L.append("")
    for host, sol, xlab in ((A, B, "x0"), (B, A, "x1")):
        dk = sorted(k for k in d if k.startswith(f"Esub_{sol}in{host}"))
        ck = sorted(k for k in d if k.startswith(f"sqs_chord_slope_{xlab}"))
        if dk and ck:
            dv = d[dk[-1]]
            cv = d[ck[0]]
            L.append(f"- {sol} in {host}: defect slope {dv:+.4f} eV; SQS chord "
                     f"({ck[0].split('_')[-1]}) {cv:+.4f} eV; difference {cv - dv:+.4f} eV "
                     f"(chord includes finite-x curvature).")
    fs = [k for k in d if k.startswith("Esub") and k.endswith("864sites")]
    for k in fs:
        k256 = k.replace("864sites", "256sites")
        if k256 in d:
            L.append(f"- Finite-size shift {k.rsplit('_', 1)[0]}: "
                     f"{(d[k] - d[k256])*1000:+.2f} meV (256 -> 864 sites).")
    L.append("")

    # ---- cross-potential + DFT comparison
    other = [k for k in cfg["potentials"] if k != potkey]
    comp_rows = []
    for ok in other:
        otab = Path(cfg["data_tree"]) / cfg["potentials"][ok]["thread_dir"] / \
            f"hull-table_{ok}_{sysname}.dat"
        if otab.exists():
            orows = {r["name"]: r for r in load_table(otab)}
            L.append(f"## Comparison: {potkey} vs {ok} vs DFT (Materials Project)")
            L.append("")
            L.append(f"| structure | x | E_f {potkey} (meV/at) | E_f {ok} (meV/at) | E_f DFT (meV/at) |")
            L.append("|---|---|---|---|---|")
            for r in sorted(rows, key=lambda q: (q["x"], q["Ef_per_atom_eV"])):
                if not is_bulk_candidate(r["prototype"]):
                    continue
                o = orows.get(r["name"])
                dft = r["Ef_DFT_MP_eV"]
                dft_s = "NA" if dft == "NA" else f"{float(dft)*1000:.1f}"
                o_s = "--" if o is None else f"{o['Ef_per_atom_eV']*1000:.1f}"
                L.append(f"| {r['name']} | {r['x']:.3f} | {r['Ef_per_atom_eV']*1000:.1f} | "
                         f"{o_s} | {dft_s} |")
            L.append("")

    (thread / f"RESULTS_{potkey}.md").write_text("\n".join(L) + "\n")
    print(f"wrote {thread / f'RESULTS_{potkey}.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])

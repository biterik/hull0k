"""Phase 1 driver: build the candidate structure pool.

Writes every candidate as <name>.lmps + <name>.yaml sidecar into
<data_tree>/<pool_dir>/. System-agnostic: everything system-specific
comes from the config YAML.

Sub-pools (brief section 3):
  a. end members: fcc/bcc/hcp of both elements
  b. standard binary prototype suite (both orderings where applicable)
  d. SQS solid solutions (icet), fcc, one per composition
  e. point defects (vacancies + substitutionals) in 4x4x4 supercells
     (+ one 6x6x6 spot check)
Sub-pool c (Materials Project) is in mp_pool.py (needs the API key).

Usage: python -m hull0k.build_pool configs/Ni-Cu.yaml [pools...]
       pools default: endmembers prototypes sqs defects

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from . import prototypes as P
from .lmps_io import write_candidate


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ref_lattice(cfg, el):
    """Reference (ground-state) lattice of element `el` for formation and defect
    energies: config `reference_lattice: {Nb: bcc}`; default fcc (the Ni-Cu
    behaviour, unchanged). Added 2026-10-07 for Ni-Nb (bcc Nb)."""
    return (cfg.get("reference_lattice") or {}).get(el, "fcc")


def vegard_a(cfg, x_B):
    A, B = cfg["elements"]
    return (1 - x_B) * cfg["a_fcc"][A] + x_B * cfg["a_fcc"][B]


def atomic_volume(cfg, x_B):
    return vegard_a(cfg, x_B) ** 3 / 4.0


def build_endmembers(cfg):
    out = []
    for el in cfg["elements"]:
        a = cfg["a_fcc"][el]
        vat = a ** 3 / 4.0
        out.append((f"fcc_{el}", P.fcc(el, a)))
        out.append((f"bcc_{el}", P.bcc(el, (2.0 * vat) ** (1.0 / 3.0))))
        out.append((f"hcp_{el}", P.hcp(el, a / np.sqrt(2.0))))
    return out


def build_prototypes(cfg):
    """Binary suite. A3B / A2B prototypes in both orderings; AB prototypes
    once (their two sublattices are exchange-equivalent)."""
    A, B = cfg["elements"]
    out = []

    def afcc(x_B):
        return vegard_a(cfg, x_B)

    def vat(x_B):
        return atomic_volume(cfg, x_B)

    for (maj, mnr, xB_A3B) in [(A, B, 0.25), (B, A, 0.75)]:
        tag = f"{maj}3{mnr}"
        a = afcc(xB_A3B)
        v = vat(xB_A3B)
        out.append((f"L12_{tag}", P.L12(maj, mnr, a)))
        out.append((f"D022_{tag}", P.D022(maj, mnr, a)))
        out.append((f"D019_{tag}", P.D019(maj, mnr, np.sqrt(2.0) * a)))
        out.append((f"D0a_{tag}", P.D0a(maj, mnr, scale=(v / 12.710) ** (1.0 / 3.0))))
        out.append((f"A15_{tag}", P.A15(maj, mnr, (8.0 * v) ** (1.0 / 3.0))))

    for (maj, mnr, xB_A2B) in [(A, B, 1.0 / 3.0), (B, A, 2.0 / 3.0)]:
        tag = f"{maj}2{mnr}"
        v = vat(xB_A2B)
        out.append((f"C11b_{tag}", P.C11b(maj, mnr, (2.0 * v) ** (1.0 / 3.0))))
        out.append((f"C15_{tag}", P.C15(maj, mnr, (24.0 * v) ** (1.0 / 3.0))))

    a50 = afcc(0.5)
    v50 = vat(0.5)
    tag = f"{A}{B}"
    out.append((f"L10_{tag}", P.L10(A, B, a50)))
    out.append((f"B2_{tag}", P.B2(A, B, (2.0 * v50) ** (1.0 / 3.0))))
    out.append((f"B1_{tag}", P.B1(A, B, (8.0 * v50) ** (1.0 / 3.0))))
    out.append((f"B3_{tag}", P.B3(A, B, (8.0 * v50) ** (1.0 / 3.0))))
    out.append((f"L11_{tag}", P.L11(A, B, a50)))
    out.append((f"Z2_{tag}", P.Z2(A, B, a50)))
    out.append((f"S40_{tag}", P.S40(A, B, a50)))
    return out


def parent_lattice(a_fcc, el, lat="fcc"):
    """One conventional cell of a parent lattice (fcc | bcc | hcp) for
    element `el`, with a volume-consistent lattice parameter derived
    from the fcc a-guess `a_fcc`. Used by the SQS and defect pools; add
    lattices here and both generalize together."""
    vat = a_fcc ** 3 / 4.0
    if lat == "fcc":
        return P.fcc(el, a_fcc), "A1_fcc"
    if lat == "bcc":
        return P.bcc(el, (2.0 * vat) ** (1.0 / 3.0)), "A2_bcc"
    if lat == "hcp":
        return P.hcp(el, a_fcc / np.sqrt(2.0)), "A3_hcp"
    raise SystemExit(f"unknown parent lattice '{lat}' (fcc|bcc|hcp)")


def build_defects(cfg):
    """Vacancies and substitutionals in supercells of the end members.
    Host lattice from config `defects.host_lattice` (default fcc)."""
    A, B = cfg["elements"]
    nx, ny, nz = cfg["defects"]["supercell"]
    lat = cfg["defects"].get("host_lattice", "fcc")
    out = []

    def supercell(el, reps):
        cell, proto = parent_lattice(cfg["a_fcc"][el], el, lat)
        at = cell.repeat(reps)
        at.info["prototype"] = f"{proto}_supercell"
        at.info["box_coupling"] = "aniso"
        return at

    for host, sol in [(A, B), (B, A)]:
        sc = supercell(host, (nx, ny, nz))
        vac = sc.copy()
        del vac[0]
        vac.info.update(prototype=f"vacancy_in_{lat}", box_coupling="aniso")
        out.append((f"vac_{lat}{host}_{nx}x{ny}x{nz}_{len(vac)}at", vac,
                    {"defect": "vacancy", "host": host, "parent_sites": len(sc)}))
        sub = sc.copy()
        sub.symbols[0] = sol
        sub.info.update(prototype=f"substitutional_in_{lat}", box_coupling="aniso")
        out.append((f"sub_{sol}in{host}_{nx}x{ny}x{nz}_{len(sub)}at", sub,
                    {"defect": "substitutional", "host": host, "solute": sol,
                     "parent_sites": len(sc)}))

    # spot check: solute-B-in-A substitutional at the larger supercell
    mx, my, mz = cfg["defects"]["spotcheck_supercell"]
    sc = supercell(A, (mx, my, mz))
    sub = sc.copy()
    sub.symbols[0] = B
    sub.info.update(prototype=f"substitutional_in_{lat}", box_coupling="aniso")
    out.append((f"sub_{B}in{A}_{mx}x{my}x{mz}_{len(sub)}at", sub,
                {"defect": "substitutional", "host": A, "solute": B,
                 "parent_sites": len(sc), "note": "finite-size spot check"}))
    # matching perfect host reference at the same size (for the finite-size shift)
    out.append((f"{lat}_{A}_{mx}x{my}x{mz}_{len(sc)}at", sc,
                {"note": f"perfect host reference for the {mx}x{my}x{mz} spot check"}))
    return out


def build_sqs(cfg):
    """SQS via icet, one per composition, on the configured parent lattice
    (`sqs.parent`: fcc | bcc | hcp, default fcc -- icet itself is
    lattice-agnostic). Records cluster-space cutoffs and achieved
    correlation mismatch vs the ideal random cluster vector."""
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    A, B = cfg["elements"]
    scfg = cfg["sqs"]
    parent = scfg.get("parent", "fcc")
    out = []
    for x_B in scfg["compositions"]:
        a = vegard_a(cfg, x_B)
        prim, _ = parent_lattice(a, A, parent)
        cs = ClusterSpace(prim, cutoffs=scfg["cutoffs"], chemical_symbols=[[A, B]] * len(prim))
        reps = round((scfg["natoms"] / len(prim)) ** (1.0 / 3.0))
        supercells = [prim.repeat(reps)]
        sqs = generate_sqs_from_supercells(
            cluster_space=cs, supercells=supercells,
            target_concentrations={A: 1 - x_B, B: x_B},
            n_steps=scfg["n_steps"],
        )
        cv = np.array(cs.get_cluster_vector(sqs))
        # ideal random cluster vector from icet itself at the ACHIEVED
        # concentration. (Until 2026-10-07 this was <s>^k with a hard-coded
        # sign convention, which icet 4.x flipped: off-stoichiometric SQS
        # then showed a spurious mismatch of 1.0 from the odd-order terms.)
        from icet.tools.structure_generation import _get_sqs_cluster_vector
        n_B = sum(1 for s in sqs.get_chemical_symbols() if s == B)
        cv_rand = np.array(_get_sqs_cluster_vector(
            cs, {A: 1 - n_B / len(sqs), B: n_B / len(sqs)}))
        mismatch = float(np.max(np.abs(np.array(cv) - cv_rand)))
        sqs.info.update(prototype=f"SQS_{parent}", box_coupling="aniso")
        name = f"sqs_{parent}_x{x_B:.2f}_{len(sqs)}at"
        out.append((name, sqs, {
            "sqs_cutoffs_A": list(scfg["cutoffs"]),
            "sqs_n_steps": scfg["n_steps"],
            "sqs_correlation_mismatch_max": round(mismatch, 6),
        }))
    return out


def main(argv):
    cfg = load_config(argv[0])
    pools = argv[1:] or ["endmembers", "prototypes", "sqs", "defects"]
    outdir = Path(cfg["data_tree"]) / cfg["pool_dir"]
    elements = cfg["elements"]
    built = []
    for pool in pools:
        if pool == "endmembers":
            items = [(n, at, {}) for n, at in build_endmembers(cfg)]
            src = "hull0k.prototypes (end member)"
        elif pool == "prototypes":
            items = [(n, at, {}) for n, at in build_prototypes(cfg)]
            src = "hull0k.prototypes (standard binary suite)"
        elif pool == "sqs":
            items = build_sqs(cfg)
            src = "icet SQS generation"
        elif pool == "defects":
            items = build_defects(cfg)
            src = "hull0k.build_pool (defect supercell)"
        else:
            raise SystemExit(f"unknown pool: {pool}")
        for name, atoms, extra in items:
            sc = write_candidate(atoms, name, outdir, elements, src, extra=extra)
            built.append(sc)
            print(f"  {name:34s} n={sc['natoms']:4d} x_{elements[1]}={sc[f'x_{elements[1]}']:.4f} "
                  f"sg={sc['spacegroup_as_built']} [{sc['box_coupling']}]")
    print(f"pool: {len(built)} candidates written to {outdir}")


if __name__ == "__main__":
    main(sys.argv[1:])

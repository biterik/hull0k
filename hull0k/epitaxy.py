"""Epitaxy models: the pluggable constraint layer of the strained hull.

A model bundles everything orientation-specific:
  - `ref(cfg, potkey, thread, elements)` -> ctx dict with the substrate
    reference (e.g. a0_ref) resolved from the phase-2 results;
  - `film(at0, side, eps, ctx)` -> (strained Atoms, record dict): takes
    the RELAXED bulk structure, applies the in-plane constraint at
    strain eps, and records every choice made;
  - `eligible` / `eligible_sqs`: prototype labels the model can handle
    (SQS listed separately so the driver can gate them with a flag);
  - `description`: one line, written into every sidecar;
  - `dir_tag`: format string naming the model's output directory, with
    `{A}`/`{B}` standing for the config's elements -- so the outputs of
    two different orientations never collide, and no directory name is
    hard-coded anywhere in the driver or the analysis.

TO ADD A NEW ORIENTATION OR SUBSTRATE (e.g. fcc(111), bcc(001) via a
Bain-type orientation relationship, an hcp substrate): write the two
functions, register them under a new name in EPITAXY_MODELS, list the
prototypes they can orient, and select the model in the config:

    epitaxy:
      model: fcc001-cube-on-cube

Nothing else changes -- strain.py, strain_hull.py and the template are
orientation-agnostic (they only fix lx, ly and relax z + internals on
whatever cell `film` returns).

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

from pathlib import Path

import yaml


def _fcc001_ref(cfg, potkey, thread, elements):
    """Substrate = this potential's relaxed fcc A; a0_ref from phase 2."""
    A = elements[0]
    with open(Path(thread) / "RELAX" / f"fcc_{A}" / "run.yaml") as f:
        r = yaml.safe_load(f)["result"]
    return {"a0_ref": float(r["cell_relaxed"]["lx"]),
            "substrate": f"fcc {A} (001)"}


def _fcc001_film(at0, side, eps, ctx):
    """Cube-on-cube (001): both in-plane cell vectors scaled so the
    film's fcc lattice constant equals (1+eps)*a0_ref. The number of
    conventional cells along x is inferred from the atomic volume
    (valid for close-packed fcc-derived cells only -- hence the
    eligibility list)."""
    aref = ctx["a0_ref"]
    at = at0.copy()
    cell = at.get_cell().array.copy()
    lx = float(at.cell.lengths()[0])
    a_est = (4.0 * at.get_volume() / len(at)) ** (1.0 / 3.0)
    ncells = max(1, round(float(lx / a_est)))
    target = ncells * (1 + eps) * aref
    cell[0, :] *= target / cell[0, 0]
    cell[1, :] *= target / cell[1, 1]
    at.set_cell(cell, scale_atoms=True)
    rec = {"eps": float(eps), "a0_ref_A": float(aref),
           "ncells_inplane": ncells,
           "orientation": "(001) cube-on-cube (choice, not optimum)",
           "bulk_inplane_a_A": float(lx) / ncells}
    return at, rec


EPITAXY_MODELS = {
    "fcc001-cube-on-cube": {
        "ref": _fcc001_ref,
        "film": _fcc001_film,
        # close-packed fcc-derived prototypes with a (001) cube-on-cube
        # assignment; open fcc-sublattice structures (B1/B3) are excluded:
        # no 1:1 commensurability with the substrate at sane strains
        "eligible": {"A1_fcc", "A1_fcc_supercell", "L1_2", "L1_0",
                     "D0_22", "Z2", "40",
                     "vacancy_in_fcc", "substitutional_in_fcc"},
        "eligible_sqs": {"SQS_fcc"},
        "description": "biaxial (001) epitaxy on fcc A, cube-on-cube; "
                       "lx = ly fixed at ncells*(1+eps)*a0_ref, lz + "
                       "internal coordinates free",
        # output directory is <thread>/STRAIN-<dir_tag>; {A}/{B} are the
        # config's elements. Override per project with `epitaxy: strain_dir:`.
        "dir_tag": "001-{A}",
    },
}


def get_model(cfg):
    name = (cfg.get("epitaxy") or {}).get("model", "fcc001-cube-on-cube")
    if name not in EPITAXY_MODELS:
        raise SystemExit(f"unknown epitaxy model '{name}'; known: "
                         f"{sorted(EPITAXY_MODELS)}")
    return name, EPITAXY_MODELS[name]


def strain_dirname(cfg, model=None):
    """Name of the strained-hull output directory under the thread dir.

    Derived from the selected epitaxy model's `dir_tag` with the config's
    elements substituted, so a second orientation writes somewhere else
    instead of overwriting the first. `epitaxy: strain_dir:` in the config
    overrides it outright. Nothing here is system-specific: for elements
    [Ni, Cu] and the shipped fcc(001) model the name is STRAIN-001-Ni,
    for [Ni, Nb] it is STRAIN-001-Ni likewise -- the tag names the
    SUBSTRATE and the orientation, not the alloy.
    """
    override = (cfg.get("epitaxy") or {}).get("strain_dir")
    if override:
        return str(override)
    if model is None:
        _, model = get_model(cfg)
    elements = cfg["elements"]
    tag = model.get("dir_tag", "epitaxy")
    return "STRAIN-" + tag.format(A=elements[0], B=elements[1])


def find_strain_dir(cfg, thread, model=None):
    """Locate an EXISTING strained-hull directory for reading.

    Returns the configured directory when it exists. Otherwise, if exactly
    one STRAIN-* directory is present (a tree written before the name became
    model-derived, or under a different model), returns that one and says so
    on stdout -- a silent guess would be worse than a loud one. With several
    candidates it refuses and names them, so the config picks.
    """
    from pathlib import Path
    thread = Path(thread)
    want = thread / strain_dirname(cfg, model)
    if want.is_dir():
        return want
    others = sorted(d for d in thread.glob("STRAIN-*") if d.is_dir())
    if len(others) == 1:
        print(f"note: {want.name} not found; using the only strained-hull "
              f"directory present, {others[0].name}")
        return others[0]
    if len(others) > 1:
        raise SystemExit(
            f"{want.name} not found and several strained-hull directories "
            f"exist: {[d.name for d in others]}. Set `epitaxy: strain_dir:` "
            f"in the config to choose one.")
    return want

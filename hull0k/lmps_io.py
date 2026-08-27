"""Writing candidate structures as LAMMPS data files + YAML sidecars.

Global type convention (brief section 1, rule 8): type 1 = first element
of the config's `elements` list (Ni), type 2 = second (Cu). Enforced here
via ase specorder; every data file gets both types declared even if one
species is absent, so the same pair_coeff line works for every candidate.

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import datetime
import numpy as np
import yaml
from ase.calculators.lammps import Prism
from ase.io.lammpsdata import write_lammps_data
import spglib


def spacegroup_of(atoms, symprec=1e-3):
    cell = (atoms.get_cell().array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        return "P1 (1)"
    return f"{ds.international} ({ds.number})"


def composition(atoms, elements):
    counts = {el: 0 for el in elements}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1
    return counts


def write_candidate(atoms, name, outdir, elements, source, extra=None, symprec=1e-3):
    """Write <name>.lmps + <name>.yaml sidecar into outdir. Returns sidecar dict."""
    outdir.mkdir(parents=True, exist_ok=True)
    lmps_path = outdir / f"{name}.lmps"
    # triclinic cells must be in LAMMPS orientation; ase handles via prismobj
    write_lammps_data(
        str(lmps_path), atoms, specorder=list(elements), masses=True,
        atom_style="atomic", prismobj=Prism(atoms.get_cell()),
    )
    counts = composition(atoms, elements)
    n = len(atoms)
    x_B = counts[elements[1]] / n if n else 0.0
    cellpar = atoms.cell.cellpar()
    sidecar = {
        "name": name,
        "prototype": atoms.info.get("prototype", "unknown"),
        "source": source,
        "composition": counts,
        "natoms": n,
        f"x_{elements[1]}": round(x_B, 8),
        "spacegroup_as_built": spacegroup_of(atoms, symprec),
        "cell_abc_A": [round(float(v), 6) for v in cellpar[:3]],
        "cell_angles_deg": [round(float(v), 4) for v in cellpar[3:]],
        "box_coupling": atoms.info.get("box_coupling", "tri"),
        "date": datetime.date.today().isoformat(),
    }
    if extra:
        sidecar.update(extra)
    with open(outdir / f"{name}.yaml", "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)
    return sidecar

"""Phase 1c: Materials Project entries for the config's chemical system.

Pulls every entry of the binary chemsys (stable + above hull), keeps the
structure, MP id and MP formation energy per atom (the DFT comparison
column), deduplicates against the already-built prototype pool with
pymatgen StructureMatcher, and writes candidates + sidecars. Duplicates
are NOT written as new candidates; instead the matching pool sidecar
gains the MP provenance + DFT numbers.

The API key is read from MATERIALS_PROJECT_API_KEY in the environment,
falling back to an interactive login shell ($SHELL, then zsh, then bash)
because rc-file exports do not reach a non-interactive process. It is
passed explicitly to MPRester, never printed, and never written to any
file.

Usage: python -m hull0k.mp_pool configs/Ni-Cu.yaml

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml
from ase.data import atomic_numbers
from ase.io import read as ase_read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from .build_pool import load_config
from .lmps_io import write_candidate


VAR = "MATERIALS_PROJECT_API_KEY"


def get_api_key():
    """Read the Materials Project key from the environment.

    Falls back to asking an INTERACTIVE login shell, because the key is
    typically exported from a shell rc file that a non-interactive process
    never sources. Tries the user's own $SHELL first, then zsh and bash, so
    this works wherever you happen to be -- no shell is assumed. The value is
    never printed and never written to any file.
    """
    key = os.environ.get(VAR, "")
    if key:
        return key
    candidates = []
    user_shell = os.environ.get("SHELL", "")
    if user_shell:
        candidates.append(user_shell)
    candidates += ["zsh", "bash"]
    seen = set()
    for sh in candidates:
        name = os.path.basename(sh)
        if name in seen:
            continue
        seen.add(name)
        try:
            out = subprocess.run([sh, "-ic", f"echo ${VAR}"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        text = out.stdout.strip()
        if text:
            key = text.splitlines()[-1].strip()
            if key:
                return key
    raise SystemExit(
        f"{VAR} not found. Export it in the environment you launch from, e.g.\n"
        f"  export {VAR}=<your-key>\n"
        f"(the key is passed explicitly to MPRester and never written to disk)")


def main(argv):
    cfg = load_config(argv[0])
    elements = cfg["elements"]
    outdir = Path(cfg["data_tree"]) / cfg["pool_dir"]
    chemsys = cfg["materials_project"]["chemsys"]

    from mp_api.client import MPRester

    with MPRester(api_key=get_api_key()) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            fields=["material_id", "formula_pretty", "structure",
                    "formation_energy_per_atom", "energy_above_hull",
                    "symmetry", "theoretical"],
        )
    print(f"MP: {len(docs)} entries for chemsys {chemsys}")

    # load existing pool structures for dedup
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(attempt_supercell=True)
    pool = {}
    for y in sorted(outdir.glob("*.yaml")):
        with open(y) as f:
            side = yaml.safe_load(f)
        lmps = outdir / f"{side['name']}.lmps"
        if not lmps.exists():
            continue
        at = ase_read(str(lmps), format="lammps-data", style="atomic",
                      Z_of_type={i + 1: atomic_numbers[el] for i, el in enumerate(elements)})
        pool[side["name"]] = (adaptor.get_structure(at), side, y)

    for doc in docs:
        mpid = str(doc.material_id)
        s = doc.structure
        match = None
        for name, (ps, side, ypath) in pool.items():
            try:
                if matcher.fit(s, ps):
                    match = (name, side, ypath)
                    break
            except Exception:
                continue
        mp_info = {
            "mp_id": mpid,
            "mp_formula": doc.formula_pretty,
            "mp_formation_energy_per_atom_eV": (None if doc.formation_energy_per_atom is None
                                                else round(float(doc.formation_energy_per_atom), 6)),
            "mp_energy_above_hull_eV": (None if doc.energy_above_hull is None
                                        else round(float(doc.energy_above_hull), 6)),
            "mp_spacegroup": str(doc.symmetry.symbol) if doc.symmetry else None,
            "mp_url": f"https://materialsproject.org/materials/{mpid}",
        }
        if match:
            name, side, ypath = match
            side.setdefault("mp_matches", []).append(mp_info)
            with open(ypath, "w") as f:
                yaml.safe_dump(side, f, sort_keys=False)
            print(f"  {mpid} {doc.formula_pretty}: duplicate of pool entry {name} -- provenance merged")
        else:
            at = adaptor.get_atoms(s)
            at.info.update(prototype=f"MP:{doc.formula_pretty}", box_coupling="tri")
            name = f"mp_{mpid.replace('mp-', '')}_{doc.formula_pretty}"
            write_candidate(at, name, outdir, elements,
                            source=f"Materials Project {mpid}", extra=mp_info)
            print(f"  {mpid} {doc.formula_pretty}: NEW candidate {name} (n={len(at)})")


if __name__ == "__main__":
    main(sys.argv[1:])

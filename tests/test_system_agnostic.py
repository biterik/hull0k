"""Regression test: nothing in the analysis layer is tied to one system.

This test exists because the tool was developed on Ni-Cu and the first
version leaked that system into places it did not belong: a hard-coded
output directory named after the substrate, a metastability table that
skipped rows by matching literal structure NAMES, an SQS consistency check
that fired only for an fcc parent lattice, and a defect analysis that
guessed the host from the composition. Every one of those worked perfectly
on Ni-Cu and would have produced wrong or missing output on Ni-Nb.

The rule being enforced: the analysis classifies an entry by its PROTOTYPE
LABEL and by what the pool builder recorded in the sidecar -- never by its
name, and never by the elements involved.

Fast, stdlib-only, and needs NO LAMMPS binary and no potential files: the
heavy third-party packages are stubbed, because none of the logic under
test uses them. Run it on every change.

Usage: python tests/test_system_agnostic.py
Exit code 0 and 'ALL TESTS PASSED' on success.

Author: Erik Bitzek (created with LLM-LMPS)
"""

import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- stub the third-party packages the imported modules pull in at import
# time. None of the code under test touches them; stubbing keeps this test
# runnable in a bare interpreter, which is the point.
for _name in ("ase", "ase.data", "ase.io", "ase.io.lammpsdata", "ase.calculators",
              "ase.calculators.lammps", "ase.spacegroup", "spglib", "scipy",
              "scipy.spatial", "pymatgen", "pymatgen.analysis",
              "pymatgen.analysis.phase_diagram", "pymatgen.core",
              "pymatgen.entries", "pymatgen.entries.computed_entries"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["ase"].Atoms = object
sys.modules["ase.data"].atomic_numbers = {"Ni": 28, "Cu": 29, "Nb": 41,
                                          "Al": 13, "Ti": 22}
sys.modules["ase.io"].read = lambda *a, **k: None
sys.modules["ase.io.lammpsdata"].write_lammps_data = lambda *a, **k: None
sys.modules["ase.calculators.lammps"].Prism = object
sys.modules["ase.spacegroup"].crystal = lambda *a, **k: None
sys.modules["spglib"].get_symmetry_dataset = lambda *a, **k: None
sys.modules["scipy.spatial"].ConvexHull = object

from hull0k import hull, relax, run_one                        # noqa: E402
from hull0k.epitaxy import find_strain_dir, get_model, strain_dirname   # noqa: E402
from hull0k.labels import (is_bulk_candidate, is_defect, is_sqs,        # noqa: E402
                           is_supercell_reference)


def test_prototype_vocabulary():
    """Classification works for every parent lattice the builders support."""
    for lattice in ("fcc", "bcc", "hcp"):
        assert is_defect(f"vacancy_in_{lattice}")
        assert is_defect(f"substitutional_in_{lattice}")
        assert is_sqs(f"SQS_{lattice}")
        assert is_bulk_candidate(f"SQS_{lattice}")
        assert not is_bulk_candidate(f"vacancy_in_{lattice}")
    for proto in ("A1_fcc_supercell", "A2_bcc_supercell", "A3_hcp_supercell"):
        assert is_supercell_reference(proto)
        assert not is_bulk_candidate(proto)
    for proto in ("A1_fcc", "L1_2", "D0_22", "40", "MP:NbNi3"):
        assert is_bulk_candidate(proto)


def test_strain_directory_is_model_derived():
    """The strained-hull directory follows the epitaxy model, not the alloy."""
    ni_cu = {"elements": ["Ni", "Cu"], "epitaxy": {"model": "fcc001-cube-on-cube"}}
    # backwards compatibility: trees written before the name was derived
    assert strain_dirname(ni_cu) == "STRAIN-001-Ni"
    # a config with no epitaxy block still resolves through the default model
    assert strain_dirname({"elements": ["Ni", "Nb"]}) == "STRAIN-001-Ni"
    # the tag names the SUBSTRATE, so a different first element moves it
    assert strain_dirname({"elements": ["Al", "Ti"]}) == "STRAIN-001-Al"
    # explicit override wins outright
    assert strain_dirname({"elements": ["Al", "Ti"],
                           "epitaxy": {"strain_dir": "STRAIN-CUSTOM"}}) == "STRAIN-CUSTOM"
    # every registered model must be able to name its own directory,
    # otherwise two orientations would overwrite each other
    for name, model in __import__("hull0k.epitaxy", fromlist=["x"]).EPITAXY_MODELS.items():
        assert "dir_tag" in model, f"model {name} has no dir_tag"

    ni_cu_model = get_model(ni_cu)[1]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "STRAIN-001-Ni").mkdir()
    assert find_strain_dir(ni_cu, tmp, ni_cu_model).name == "STRAIN-001-Ni"

    legacy = Path(tempfile.mkdtemp())
    (legacy / "STRAIN-LEGACY-NAME").mkdir()
    assert find_strain_dir(ni_cu, legacy).name == "STRAIN-LEGACY-NAME"

    ambiguous = Path(tempfile.mkdtemp())
    (ambiguous / "STRAIN-A").mkdir()
    (ambiguous / "STRAIN-B").mkdir()
    try:
        find_strain_dir(ni_cu, ambiguous)
    except SystemExit as exc:
        assert "strain_dir" in str(exc)
    else:
        raise AssertionError("ambiguous tree should refuse, not guess")


def test_defect_energies_use_the_sidecar_not_the_composition():
    """Host and solute come from what the pool builder recorded.

    Built on a bcc-hosted Ni-Nb set, so an fcc-only or composition-guessing
    implementation fails here even though it passes on dilute fcc Ni-Cu.
    """
    cfg = {"elements": ["Ni", "Nb"]}
    E_Ni, E_Nb = -4.45, -7.57
    entries = [
        {"name": "vac_bccNb", "prototype": "vacancy_in_bcc", "defect": "vacancy",
         "host": "Nb", "solute": None, "n": 53, "x": 1.0, "Ef": 0.01,
         "E_total": 53 * E_Nb + 2.7},
        {"name": "sub_NiinNb", "prototype": "substitutional_in_bcc",
         "defect": "substitutional", "host": "Nb", "solute": "Ni", "n": 54,
         "x": 53 / 54, "Ef": 0.004, "E_total": 53 * E_Nb + E_Ni + 0.35},
        {"name": "sqs_bcc_x0.50", "prototype": "SQS_bcc", "defect": None,
         "host": None, "solute": None, "n": 54, "x": 0.5, "Ef": 0.06,
         "E_total": 0.0},
    ]
    d = hull.defect_energies(entries, cfg, E_Ni, E_Nb)
    assert abs(d["Evac_Nb_54sites"] - 2.7) < 1e-9, d
    assert abs(d["Esub_NiinNb_54sites"] - 0.35) < 1e-9, d
    assert "dilute_slope_NiinNb_54sites" in d
    # the SQS chord slopes are the dilute-limit consistency check; an
    # implementation keyed to the literal prototype "SQS_fcc" produces none
    assert any(k.startswith("sqs_chord_slope") for k in d), d


def test_report_spectrum_filters_by_prototype():
    """Defect cells and finite-size references leave the spectrum table by
    LABEL, so the filter survives new elements, lattices and cell sizes."""
    rows = [
        ("fcc_Nb", "A1_fcc", True),
        ("fcc_Nb_6x6x6_864at", "A1_fcc_supercell", False),
        ("bcc_Nb_5x5x5_250at", "A2_bcc_supercell", False),
        ("vac_bccNb_4x4x4_127at", "vacancy_in_bcc", False),
        ("sub_NiinNb_4x4x4_128at", "substitutional_in_bcc", False),
        ("L12_Ni3Nb", "L1_2", True),
        ("sqs_bcc_x0.50_54at", "SQS_bcc", True),
    ]
    for name, proto, expected in rows:
        assert is_bulk_candidate(proto) is expected, (name, proto)


def test_pressure_gate_is_coupling_aware_and_shared():
    """The two phase-2 runners must gate on the same pressure components.

    This is the regression test for a bug that was invisible on `iso` cells:
    the integrated runner exited its fallback loop on the SCALAR pressure
    while the decoupled runner used the per-component gate. An `aniso` or
    `tri` cell can average to zero pressure with large opposing components,
    so the two paths took different numbers of fallback cycles and stopped
    at different structures -- breaking the bit-identical guarantee the
    whole decoupled workflow rests on.

    Needs no LAMMPS: the gate is a pure decision over a parsed result dict.
    """
    # a cell that is relaxed ON AVERAGE but not component-wise: the exact
    # case that separates the scalar test from the correct one
    res = {"press_bar": 0.0, "pxx_bar": +5000.0, "pyy_bar": -5000.0,
           "pzz_bar": 0.0, "pxy_bar": 0.0, "pxz_bar": 0.0, "pyz_bar": 0.0}
    assert relax.press_gate(res, "iso"), "iso gates on the scalar only"
    assert not relax.press_gate(res, "aniso"), "aniso must see pxx/pyy"
    assert not relax.press_gate(res, "tri"), "tri must see pxx/pyy"
    # the scalar pressure alone would have passed all three -- that was the bug
    assert abs(res["press_bar"]) < relax.PRESS_TOL_BAR

    # a genuinely relaxed cell passes every coupling
    good = {f"{k}_bar": 0.0 for k in
            ("press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")}
    for coupling in ("iso", "aniso", "tri"):
        assert relax.press_gate(good, coupling)

    # unresolved shear fails tri but not aniso, which does not relax it
    sheared = dict(good, pxy_bar=+800.0)
    assert relax.press_gate(sheared, "aniso")
    assert not relax.press_gate(sheared, "tri")

    # and the decoupled runner, driven by what prepare.py bakes into
    # job.json, must return the SAME verdict for every case and coupling
    for case in (res, good, sheared):
        for coupling in ("iso", "aniso", "tri"):
            job = {"press_tol_bar": relax.PRESS_TOL_BAR,
                   "press_gate_components": relax.press_gate_components(coupling)}
            assert run_one.press_ok(job, case) == relax.press_gate(case, coupling), (
                f"runners disagree for coupling {coupling} on {case}")

    # every coupling the pool builder can assign must have a gate
    from hull0k import prototypes  # noqa: F401
    for coupling in ("iso", "aniso", "tri"):
        assert relax.press_gate_components(coupling)
    try:
        relax.press_gate_components("nonsense")
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown coupling must fail loudly, not silently")


def test_convergence_criterion_is_shared():
    """Both runners must also agree on when a minimisation has converged."""
    cases = [
        {"stopping_criterion": "force tolerance", "fnorm_per_dof_eV_A": 1e-3},
        {"stopping_criterion": "max iterations", "fnorm_per_dof_eV_A": 1e-12},
        {"stopping_criterion": "max iterations", "fnorm_per_dof_eV_A": 1e-3},
        {"stopping_criterion": None, "fnorm_per_dof_eV_A": 1e-12},
    ]
    job = {"ftol_per_dof": relax.FTOL_PER_DOF}
    for res in cases:
        # relax.py's `converged` is a closure inside relax_one; replicate its
        # published contract and check run_one agrees with it
        expected = ("force tolerance" in (res.get("stopping_criterion") or "").lower()
                    or res.get("fnorm_per_dof_eV_A", 1.0) < relax.FTOL_PER_DOF)
        assert run_one.converged(job, res) == expected, res


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__}: PASS")
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

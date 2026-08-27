"""End-to-end test of the hull0k pipeline with a FAKE potential.

Runs the real pipeline -- pool build, LAMMPS relaxation (box/relax +
fallback + equivalence gate), hull construction with the pymatgen/scipy
cross-check -- on a Lennard-Jones binary, so no real potential files are
needed. Needs only the lmp binary from the config given (ML-PACE not
required; lj/cut is in every LAMMPS).

A NOTE ON THE FAKE POTENTIAL, because it caused an afternoon of confusion:
`lj/cut` is truncated, not smoothly damped, so the lattice energy is
DISCONTINUOUS in the lattice constant -- every time a neighbour shell
crosses the cutoff radius, E jumps by tens of meV/atom and the pressure by
thousands of bar. If the starting lattice constant sits on the far side of
such a step from the equilibrium one, `fix box/relax` cannot cross it: the
line search refuses to go uphill through the discontinuity and the run
lands in the two-stage fallback and exits UNCONVERGED. That is a property
of truncated LJ, not a defect in the relaxation engine -- real EAM and ACE
potentials go smoothly to zero at their cutoff and do not do this.

The `a_fcc` values below are therefore the TRUNCATED-LJ fcc equilibria for
these (eps, sigma, rc), not textbook LJ values, so that the two fcc end
members start on the correct smooth branch. Structures whose own shell
geometry still straddles a step legitimately take the fallback, and the
test reports which ones did.

Usage: .venv/bin/python tests/test_fake_end_to_end.py
Exit code 0 and 'ALL TESTS PASSED' on success.

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hull0k.build_pool import build_endmembers, build_prototypes, load_config  # noqa: E402
from hull0k.lmps_io import write_candidate  # noqa: E402
from hull0k import relax, hull, prepare, execute, harvest  # noqa: E402


def main():
    with open(REPO / "configs/Ni-Cu.yaml") as f:
        real = yaml.safe_load(f)
    tmp = Path(tempfile.mkdtemp(prefix="hull0k-test-"))
    cfg = {
        "system": "FakeA-FakeB",
        "elements": ["Ni", "Cu"],          # type map exercised; LJ params are fake
        "masses": [58.6934, 63.546],
        # truncated-lj/cut fcc equilibria for (eps, sigma) below at rc = 6.0;
        # see the module docstring -- these are NOT the textbook LJ values
        "a_fcc": {"Ni": 3.564, "Cu": 3.805},
        "data_tree": str(tmp),
        "pool_dir": "POOL",
        "lmp_binary": real["lmp_binary"],
        "potentials": {
            "LJ-fake": {
                "thread_dir": "01_HULL-LJ-fake",
                "file": "(none)", "sha256": "(none)",
                "pair_style": "lj/cut 6.0",
                "pair_coeff": ("1 1 0.30 2.30\npair_coeff 2 2 0.25 2.45"),
                "citation": "fake LJ for pipeline test",
            }
        },
    }
    cfgpath = tmp / "config.yaml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)

    pool = tmp / "POOL"
    names = []
    for name, at in build_endmembers(cfg):
        if name.startswith(("fcc", "hcp")):   # hcp exercises the tri/triclinic path
            write_candidate(at, name, pool, cfg["elements"], "test")
            names.append(name)
    for name, at in build_prototypes(cfg):
        if name.startswith(("L12_", "B2_", "L10_")):
            write_candidate(at, name, pool, cfg["elements"], "test")
            names.append(name)
    print(f"pool: {names}")

    summaries = []
    for name in names:
        s = relax.relax_one(cfg, "LJ-fake", name)
        assert s["status"] == "OK", f"{name} relax failed: {s}"
        assert s["equiv_ok"], f"{name} equivalence gate failed: {s['equiv_gate_eV_per_atom']}"
        assert s["min_dist_ok"], f"{name} min-dist failed"
        summaries.append(s)
        print(f"  {name}: E/at = {s['E_per_atom_eV']:.6f}, path = {s['path']}")
    fallback = [s["name"] for s in summaries if "fallback" in s["path"]]
    unconverged = [s["name"] for s in summaries if "UNCONVERGED" in s["path"]]
    print(f"took the fallback: {fallback or 'none'}")
    print(f"ended UNCONVERGED: {unconverged or 'none'}  "
          f"(expected for cells whose neighbour shells straddle the LJ cutoff)")
    # The two fcc end members define every formation energy in the hull, so
    # they must relax cleanly. If this fires, the likely cause is the LJ
    # cutoff discontinuity described in the module docstring -- check that
    # cfg["a_fcc"] still matches the truncated-LJ equilibrium for this rc.
    for ref in (f"fcc_{cfg['elements'][0]}", f"fcc_{cfg['elements'][1]}"):
        s = next(q for q in summaries if q["name"] == ref)
        assert "UNCONVERGED" not in s["path"], (
            f"{ref} did not converge (path={s['path']}). The hull references "
            f"must relax cleanly; see this file's docstring on the lj/cut "
            f"cutoff discontinuity and check a_fcc[{ref.split('_')[1]}].")
        assert s["press_ok"], f"{ref} failed the pressure gate: {s['press_components_bar']}"
    thread = tmp / "01_HULL-LJ-fake"
    with open(thread / "relax-summary_LJ-fake.yaml", "w") as f:
        yaml.safe_dump(summaries, f, sort_keys=False)

    entries = hull.collect_entries(cfg, "LJ-fake")
    EA, EB = hull.formation_energies(entries, cfg)
    pd, stable = hull.hull_pymatgen(entries, cfg)
    sv = hull.hull_scipy(entries)
    assert sorted(stable) == sorted(sv), f"hull cross-check FAILED: {stable} vs {sv}"
    for e in entries:
        if e["name"] in (f"fcc_{cfg['elements'][0]}", f"fcc_{cfg['elements'][1]}"):
            assert abs(e["Ef"]) < 1e-12, f"end member Ef != 0: {e}"
    hull_pts, segs = hull.mu_curves(entries, cfg, EA, EB)
    assert len(segs) >= 1
    print(f"hull members: {stable}; mu segments: {len(segs)}")

    # --- decoupled path (prepare -> execute -> harvest) must reproduce
    # the integrated runner bit-for-bit
    # One structure per box-coupling class. This is the regression test for
    # the fallback exit criterion: iso cells cannot distinguish the scalar
    # pressure from the per-component gate, so an iso-only comparison would
    # not notice the two runners disagreeing on when to stop cycling.
    sub = ["fcc_Ni",      # iso
           "L10_NiCu",    # aniso  -- pxx/pyy/pzz gate, scalar press differs
           "hcp_Ni"]      # tri    -- all six components gated
    cfgpath2 = tmp / "config-decoupled.yaml"
    cfg2 = dict(cfg)
    cfg2["potentials"] = {"LJ-fake": dict(cfg["potentials"]["LJ-fake"],
                                          thread_dir="02_HULL-LJ-fake-DECOUPLED")}
    with open(cfgpath2, "w") as f:
        yaml.safe_dump(cfg2, f)
    prepare.main([str(cfgpath2), "LJ-fake"] + sub)
    execute.main([str(cfgpath2), "LJ-fake"] + sub)
    harvest.main([str(cfgpath2), "LJ-fake"] + sub)
    with open(tmp / "02_HULL-LJ-fake-DECOUPLED" / "relax-summary_LJ-fake.yaml") as f:
        dec = {s["name"]: s for s in yaml.safe_load(f)}
    ints = {s["name"]: s for s in summaries}
    for n in sub:
        dE = abs(dec[n]["E_total_eV"] - ints[n]["E_total_eV"])
        assert dE < 1e-10, (f"decoupled path deviates for {n} "
                            f"(coupling {ints[n]['coupling']}): dE = {dE}")
        assert dec[n]["equiv_ok"], f"decoupled equivalence gate failed for {n}"
        # the two runners must also agree on HOW they got there: same
        # fallback decisions, same number of cycles, same winning stage
        assert dec[n]["path"] == ints[n]["path"], (
            f"{n} (coupling {ints[n]['coupling']}): integrated runner took "
            f"'{ints[n]['path']}' but the decoupled path took "
            f"'{dec[n]['path']}' -- the two fallback exit criteria disagree")
    print("decoupled path reproduces integrated runner (energy AND path) for "
          + ", ".join(f"{n} [{ints[n]['coupling']}]" for n in sub))
    shutil.rmtree(tmp)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

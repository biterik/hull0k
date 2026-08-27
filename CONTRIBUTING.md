# Contributing to hull0k

Thanks for looking. `hull0k` is a small, deliberately opinionated research tool;
these notes describe the conventions that keep it usable rather than a formal
process. Issues and pull requests are welcome.

## The design rules that must not be broken

These are the load-bearing decisions. A change that violates one of them is not a
refactor, it is a different tool.

1. **LAMMPS is the only energy engine.** ASE builds structures, applies
   constraints, orchestrates and parses. Every energy, force and minimisation
   comes from a real `lmp` binary running a real input script. Never add a
   Python-side calculator or an ASE minimiser, not even "just for a quick check" —
   the whole value proposition is that any number quoted downstream is a LAMMPS
   number, guaranteed by construction rather than by comparison.

   **And it is invoked as a SUBPROCESS, never linked.** Do not replace
   `subprocess.run([lmp, ...])` with `from lammps import lammps`, however much
   faster it looks: that module links libLAMMPS in-process, LAMMPS is GPL-2.0,
   and a distributed combined work would then fall under the GPL rather than
   this project's BSD-3-Clause. The subprocess boundary is load-bearing for the
   licence as well as for the "results are LAMMPS results" guarantee. Same rule
   for dependencies: import `ase` (LGPL) and `icet` (MPL), never vendor their
   source files into this tree.

2. **The config is the system.** No element, mass, lattice parameter, file path or
   system-specific threshold belongs in `hull0k/`. If your change needs one, it
   needs a config key. A new binary system must remain zero lines of code.

   The corollary, and the rule that is easiest to break by accident: **classify
   entries by prototype label and sidecar fields, never by name.** A test like
   `name.startswith("vac_")`, or a literal `"SQS_fcc"`, or a hard-coded output
   directory named after a substrate, all work perfectly on the system you are
   developing on and silently mis-handle the next one. `hull0k/labels.py` is the
   single place that knows the prototype vocabulary; use it.
   `tests/test_system_agnostic.py` enforces this against a bcc-hosted Ni-Nb
   fixture — if you add a classification rule, add a case there.

3. **The hull-table schema is a contract.** `hull-table_<pot>_<system>.dat` is
   consumed by downstream projects. Extend it at the right-hand end; never
   reorder, rename or drop a column.

4. **The two phase-2 paths must stay identical.** `relax.py` (integrated) and
   `prepare`/`run_one`/`harvest` (decoupled) must produce bit-identical energies,
   take the same fallback decisions, and write identically-shaped
   `relax-summary` records. Any convergence or gate criterion must be written
   once and applied by both -- `relax.py` and `run_one.py` diverging on when to
   stop cycling is exactly the bug this rule exists to prevent, and it is
   invisible on `iso` cells. `tests/test_fake_end_to_end.py` compares one
   structure per box-coupling class (`iso`, `aniso`, `tri`) on both energy and
   path; keep it passing, and if you add a coupling class, add it there.

5. **`run_one.py` is stdlib-only.** It runs on cluster compute nodes with a plain
   `python3` and no virtual environment. No numpy, no ASE, no yaml, no config
   file. Everything that needs a dependency belongs in `harvest.py`.

6. **Record choices as choices.** Anything that is an assumption rather than a
   result — an orientation relationship, a reservoir convention, a skipped
   candidate — gets written into the output with a reason, in the file where a
   reader will meet it.

7. **Never silently truncate or drop.** A clipped plot states what was clipped and
   the true maximum. A skipped structure is written to a `skipped-*.yaml` with its
   reason. An unexecuted structure appears as `NOT-EXECUTED`, not as an absence.

## File and output conventions

- **Never commit an interatomic potential.** They are referenced by absolute
  path + sha256 from the config and nothing else. They are large, they carry
  their own licences and citation requirements (not this project's BSD-3-Clause),
  and some are unpublished. `.gitignore` blocks the usual extensions and the
  `ACE_*.yaml` naming, but ACE parameterisations are plain YAML and cannot be
  blanket-ignored — so the rule matters more than the filter. Same for anything
  out of the data tree: `.lmps`, `.dump` and run logs stay out.
- Plain files only: Markdown, YAML, space-separated `.dat`. **Never CSV.**
- Every non-LAMMPS `.dat` file carries `#` header lines naming each column with
  its units, plus the generating date, the potential and its SHA-256.
- Descriptive names always. LAMMPS data files end `.lmps`; dumps end `.dump`.
  No `test.in`, `run1/`, `data.lammps`, `dump.out`.
- Directories in CAPS; element symbols in canonical case (`Ni-Cu`, not `NI-CU`).
- Every LAMMPS input carries the standard header: filename, one-line purpose,
  project/thread/run, author, date.

## LAMMPS input conventions

- Stopping criterion is `fnorm/sqrt(3N) < 1e-8 eV/Å`, computed **in the input**
  from `count(all)` — LAMMPS' `ftol` is the raw, size-dependent global norm.
  Report both `fnorm/N` and `fnorm/sqrt(3N)`; never a bare `fnorm`.
- `min_style cg`. Never FIRE or quickmin with `fix box/relax`.
- `min_modify line quadratic` whenever `fix box/relax` is active.
- `write_data … nocoeff`, always, in any template whose output is re-read.
- Never override `write_data`'s default precision — the equivalence gate depends
  on the round trip.
- In `# NEED TO SET:` comments, write placeholder names in **lowercase**. The
  renderer replaces exact upper-case tokens, including ones inside comments, and a
  multi-line value substituted into a comment escapes it and executes.

## Adding things

- **A prototype** → a builder in `hull0k/prototypes.py` returning an `ase.Atoms`
  with `.info["prototype"]` and `.info["box_coupling"]`, listed in
  `build_prototypes()`. Get the coupling class right (`iso` cubic, `aniso`
  tetragonal/orthorhombic, `tri` hexagonal/trigonal/triclinic — `aniso` silently
  breaks `a = b` on a hexagonal cell) and check the sidecar's
  `spacegroup_as_built` against what you intended.
- **An epitaxy orientation** → a registry entry in `hull0k/epitaxy.py`: a `ref`
  function, a `film` function, an `eligible` set, an `eligible_sqs` set and a
  one-line `description`. Do not branch in `strain.py` or `strain_hull.py`; they
  are orientation-agnostic on purpose.
- **A parent lattice** for SQS or defects → extend `parent_lattice()` in
  `hull0k/build_pool.py`; both sub-pools generalise together.
- **A batch system** → a submission template alongside `SBATCH_TEMPLATE` in
  `hull0k/prepare.py`. The contract is small: set an array index, read the *n*-th
  line of `manifest.txt`, call `python3 run_one.py <name>`. Keep every
  site-specific value an `EDITME_` token.

## Before opening a pull request

```bash
python tests/test_system_agnostic.py     # instant; no LAMMPS, no dependencies
python tests/test_fake_end_to_end.py     # needs an lmp binary, no potential files
```

Both must print `ALL TESTS PASSED` and exit 0.

`test_system_agnostic.py` drives the classification and analysis logic with a
bcc-hosted Ni-Nb fixture; it stubs the heavy third-party packages, so it runs in a
bare interpreter and there is no excuse for skipping it.

`test_fake_end_to_end.py` exercises the pool builder, the box-relax and triclinic
paths, the two-stage fallback, the equivalence gate, the pymatgen/scipy hull
cross-check, and the decoupled-vs-integrated equivalence.

If you touched anything that changes numbers, say so explicitly in the pull
request and show a before/after on at least one structure. Energies here are
compared at the `1e-8 eV/atom` level; a change that moves them is a change in the
physics, not in the code style.

## Style

Plain, readable Python. Standard library plus the declared dependencies. Every
module opens with a docstring saying what phase it belongs to, what it reads, what
it writes and how to invoke it — keep that up to date, because it is the first
thing anyone reads.

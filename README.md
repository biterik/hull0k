# hull0k — 0 K convex hulls for binary alloys from LAMMPS potentials

**`hull0k`** builds the 0 K (static, fully minimized) formation-energy convex hull
of a binary alloy system using an interatomic potential as the only energy engine.
For every candidate structure it reports the formation energy `E_f(x)` and the
distance to the hull `e_above_hull` — the standard metastability measure used by
the DFT databases (Materials Project, OQMD, AFLOW), but evaluated with **one
classical or machine-learned potential**, so there is no reference-energy or
mixing problem: end members, ordered compounds, solid solutions and point defects
are all computed with the same Hamiltonian.

It also computes the **strained (epitaxial) hull**: the same construction under a
biaxial in-plane constraint, which answers "what is stable *on this substrate*"
and exposes strain-driven changes in hull membership.

```
                pool                relax                 hull                  strain
 config  ──►  prototypes  ──►  LAMMPS box/relax  ──►  E_f(x), e_above_hull  ──►  E_f(x, ε)
   │          SQS (icet)        integrity gates        μ(x), defect energies      coherent hull
   │          defects           equivalence gate       tables + plots             stability map
   │          Materials Project
   └── everything system-specific lives in ONE YAML file
```

**Status:** working, validated end-to-end on Ni–Cu with two potentials
(an EAM and an ACE parameterisation). See [Validation status](#12-validation-status).

---

## Table of contents

1. [What hull0k does](#1-what-hull0k-does)
2. [Installation](#2-installation)
3. [Repository layout](#3-repository-layout)
4. [Core concepts](#4-core-concepts)
5. [Manual — command reference](#5-manual--command-reference)
6. [Output formats (the interchange contract)](#6-output-formats-the-interchange-contract)
7. [Running on an HPC cluster](#7-running-on-an-hpc-cluster)
8. [Walked-through example: Ni–Cu with an EAM potential](#8-walked-through-example-nicu-with-an-eam-potential)
9. [Extending hull0k](#9-extending-hull0k)
10. [Not yet implemented / known limitations](#10-not-yet-implemented--known-limitations)
11. [Implementation tips and LAMMPS pitfalls](#11-implementation-tips-and-lammps-pitfalls)
12. [Validation status](#12-validation-status)
13. [Citing, licence, author](#13-citing-licence-author)

---

## 1. What hull0k does

### The science

For a binary A–B system, with `x = N_B / (N_A + N_B)` counting **atoms only**
(a vacancy changes `N` and therefore `x`; it is not a species), the formation
energy per atom is

```
E_f(x) = E(x) − x·E_B − (1−x)·E_A
```

where `E_A` and `E_B` are **this potential's own relaxed fcc end-member energies**.
The lower convex hull of the `(x, E_f)` point cloud gives, for every candidate:

- `e_above_hull` — how metastable it is,
- the **decomposition** it would break up into,
- the **chemical potentials** `μ_A(x)`, `μ_B(x)` and `Δμ = μ_B − μ_A` from the
  tangent construction on each hull segment,
- **defect energies** (vacancy formation, dilute substitution) referenced to the
  hull's own reservoirs, with the convention stated explicitly in every output.

The hull is built twice — once with `pymatgen`'s `PhaseDiagram` and once with a
direct `scipy.spatial.ConvexHull` lower hull — and the two vertex sets must agree.
A disagreement is a hard failure, not a warning.

For the **strained hull**, the film's in-plane cell vectors are pinned to
`(1+ε)·a₀_ref` of a chosen substrate while `lz` and all internal coordinates relax.
Both reference conventions are computed and written side by side:

| convention | reference | reads as |
|---|---|---|
| `Ef_unstrained_ref` | unstrained bulk end members | contains all elastic energy, end members included |
| `Ef_coherent_ref` | end members strained *the same way* | the **coherent hull**: what is stable on this substrate |

The two differ by a function linear in `x`, so hull membership and `e_above_hull`
are provably identical for both. `hull0k` asserts this at runtime and writes one
`e_above_hull` column.

### The guarantee: LAMMPS is the only energy engine

This is the design rule the whole package is built around.

> ASE builds structures, applies constraints, orchestrates runs and parses output.
> **Every energy, force and minimisation is computed by a real `lmp` binary running
> a real LAMMPS input script**, loading the real potential file with `units metal`.
> No Python-side calculator, no ASE minimiser, ever.

"Results identical to LAMMPS" is therefore true by construction rather than by
comparison. Two mechanisms make it auditable:

- **The equivalence gate.** After every relaxation, an *independent* single-point
  LAMMPS run (`run 0`, separate input file) re-reads the written `.lmps` file and
  must reproduce the minimiser's final energy to `< 1e-8 eV/atom`. This closes the
  loop over file I/O and proves that any number quoted downstream in Python is a
  LAMMPS number. In the Ni–Cu validation all 80 runs returned **exactly 0.0**.
- **Provenance per run.** Each run directory carries a `run.yaml` recording the
  potential file, its SHA-256, the LAMMPS command and version, the template
  version and the date.

### Convergence convention

Minimisation stops at `fnorm/sqrt(3N) < 1e-8 eV/Å`, where `fnorm` is the 2-norm of
the global 3N force vector. Because LAMMPS' `ftol` is the *raw* global norm — and
therefore size-dependent — the per-degree-of-freedom threshold is converted inside
the input script from the atom count:

```lammps
variable n_atoms       equal count(all)
variable sqrt_3N       equal sqrt(3.0*v_n_atoms)
variable FTOL_PER_DOF  equal 1.0e-8
variable ftol_global   equal v_FTOL_PER_DOF*v_sqrt_3N
minimize 0.0 ${ftol_global} MAXITER MAXEVAL
```

Both `fnorm/N` and `fnorm/sqrt(3N)` are reported everywhere; a bare `fnorm` never is.

### Integrity gates (every structure, every potential)

| gate | criterion | on failure |
|---|---|---|
| convergence | `fnorm/sqrt(3N) < 1e-8 eV/Å` | two-stage fallback, then flagged `UNCONVERGED` |
| pressure | `\|P\| < 100 bar` on every component the coupling relaxes | fallback; residual recorded |
| symmetry | spglib symbol before vs after (`symprec = 1e-3 Å`) | flagged `drifted = 1`; energy belongs to the drift product |
| minimum distance | `d_min > 1.8 Å` | flagged, entry marked not-assured |
| equivalence | independent single point matches to `< 1e-8 eV/atom` | flagged, entry marked not-assured |

Entries failing any gate are still written to the hull table (nothing is hidden),
but are plotted as **open** rather than filled markers and marked in the report.

---

## 2. Installation

### 2.1 Python environment

Python **3.11+** (developed and validated on 3.14). Create a virtual environment
inside the repository and install the package:

```bash
cd /path/to/CONVEX-HULL-0K
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e .
```

> **Install editable (`-e`).** The LAMMPS templates and the configs live beside
> the package, not inside it, and are resolved relative to the repository root.
> An editable install keeps that root findable; a plain wheel install would not.
> You can also skip installation entirely and run `python -m hull0k.<module>`
> from the repository root with the dependencies on your path.

Dependencies (all from PyPI): `ase`, `pymatgen`, `spglib`, `numpy`, `scipy`,
`matplotlib`, `pyyaml`, plus two optional extras:

```bash
./.venv/bin/pip install -e ".[sqs]"   # icet — needed only for hull0k.build_pool ... sqs
./.venv/bin/pip install -e ".[mp]"    # mp-api — needed only for hull0k.mp_pool
```

A pinned, known-good set of versions from the validation run is in
`requirements.lock`; use it if you need to reproduce the published numbers exactly:

```bash
./.venv/bin/pip install -r requirements.lock
```

### 2.2 LAMMPS

You need an `lmp` binary that includes the packages your potentials require:

| potential type | required LAMMPS package |
|---|---|
| EAM (`eam/alloy`, `eam/fs`, …) | `MANYBODY` (built in by default) |
| ACE (`pair_style pace`) | `ML-PACE` |
| SNAP, MLIAP | `ML-SNAP`, `ML-IAP` |
| the bundled test (`lj/cut`) | none — every build has it |

A serial build is entirely adequate: these are seconds-long, single-core jobs.
The build used for the validation:

```bash
cmake -S /path/to/lammps/cmake -B /tmp/lmp-build \
  -G Ninja -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=off -D BUILD_OMP=off \
  -D PKG_MANYBODY=on -D PKG_ML-PACE=on -D PKG_EXTRA-COMPUTE=on
ninja -C /tmp/lmp-build -j 8
cp /tmp/lmp-build/lmp ~/bin/lmp-pace
```

> **Check `src/version.h`, not the directory name.** A tree named
> `lammps-stable_22Jul2025_update4` turned out to be a git checkout on `develop`
> reporting "30 Mar 2026 – Development". The directory name is not the version.

### 2.3 Verify the installation

Two tests ship. The first needs nothing at all — no LAMMPS, no third-party
packages, no potential files — and takes a fraction of a second:

```bash
python tests/test_system_agnostic.py
```

It is a regression test for system-agnosticism: it drives the classification and
analysis logic with a **bcc-hosted Ni–Nb** fixture, so any code that quietly
assumes fcc, or Ni–Cu, or a particular structure name, fails here instead of
producing wrong output on your system.

The second runs the *real* pipeline on a fake Lennard-Jones binary, so it needs an
`lmp` binary but **no real potential files**:

```bash
./.venv/bin/python tests/test_fake_end_to_end.py
```

It exercises the pool builder, the box-relax path, the triclinic path, the
two-stage fallback, the equivalence gate, the pymatgen/scipy hull cross-check, and
asserts that the decoupled `prepare → execute → harvest` workflow reproduces the
integrated runner **bit-for-bit — in energy and in the fallback path taken —**
for one structure of each box-coupling class (`iso`, `aniso`, `tri`). On success
it prints `ALL TESTS PASSED` and exits 0.

> Expect some LJ structures to report `twostage-fallback-UNCONVERGED`, and the
> test to say which. That is the fixture, not the engine: `lj/cut` is truncated
> rather than smoothly damped, so the lattice energy jumps by tens of meV/atom
> every time a neighbour shell crosses the cutoff radius. A cell whose starting
> lattice constant sits on the far side of such a step from its equilibrium one
> cannot get there — the line search will not climb through a discontinuity. Real
> EAM and ACE potentials go smoothly to zero at their cutoff and do not do this.
> The two fcc end members are pinned to the truncated-LJ equilibria and the test
> *does* require those to relax cleanly, since they define every formation energy.

The test reads `lmp_binary` from `configs/Ni-Cu.yaml`; point that at your own
binary (or pass a config of your own) before running it on a fresh machine.

---

## 3. Repository layout

```
CONVEX-HULL-0K/
├── hull0k/                        the package — no system-specific values anywhere
│   ├── cli.py                     `hull0k <subcommand> …` console entry point
│   ├── prototypes.py              structure builders (explicit cells + decorated-fcc)
│   ├── lmps_io.py                 .lmps writer + YAML provenance sidecars + spglib
│   ├── build_pool.py              phase 1: end members, prototypes, SQS, defects
│   ├── mp_pool.py                 phase 1c: Materials Project entries + dedup
│   ├── relax.py                   phase 2, INTEGRATED runner (render+run+check, one process)
│   ├── prepare.py                 phase 2, DECOUPLED step 1: render everything, no LAMMPS
│   ├── run_one.py                 stdlib-only per-structure driver (staged to the cluster)
│   ├── execute.py                 phase 2, DECOUPLED step 2: run locally
│   ├── harvest.py                 phase 2, DECOUPLED step 3: checks + summary
│   ├── verify.py                  perturbed-restart verification of surprising entries
│   ├── hull.py                    phase 3: hull, μ(x), defect energies, tables, plots
│   ├── report.py                  phase 3: RESULTS_<potential>.md
│   ├── labels.py                  prototype-label vocabulary (how entries are classified)
│   ├── epitaxy.py                 phase 4: PLUGGABLE epitaxy model registry
│   ├── strain.py                  phase 4: epitaxial relaxation driver
│   └── strain_hull.py             phase 4: strained-hull analysis + stability map
├── templates/
│   ├── minimize-boxrelax.in.skel    cg + fix box/relax, coupling by symmetry
│   ├── minimize-fixedcell.in.skel   fixed-cell stage of the two-stage fallback
│   ├── minimize-epitaxial.in.skel   fix box/relax z 0.0 (lx, ly pinned)
│   └── singlepoint.in.skel          run 0 — the equivalence gate
├── configs/
│   ├── EXAMPLE.yaml               fully commented template — start here
│   └── Ni-Cu.yaml                 the validation system, as actually run
├── tests/
│   ├── test_fake_end_to_end.py    LJ pipeline test; needs lmp, no potentials
│   └── test_system_agnostic.py    system-agnosticism regression test; stdlib only
├── pyproject.toml
├── requirements.lock              pinned versions from the validation run
├── CONTRIBUTING.md
├── CITATION.cff
└── LICENSE
```

Templates use **ALL-CAPS placeholders** substituted by a plain string replace.
There is no template engine and no dependency on one.

---

## 4. Core concepts

### 4.1 The config *is* the system

Nothing in `hull0k/` knows about Ni, Cu, or any element. Everything
system-specific — elements, masses, lattice-parameter guesses, potentials, paths,
SQS parent lattice, defect host lattice, epitaxy model, validation references —
lives in one YAML file. **A new binary system is a new config file and zero lines
of code.** Start from `configs/EXAMPLE.yaml`.

### 4.2 The global type convention

> **Atom type 1 = `elements[0]`, atom type 2 = `elements[1]`, in every data file
> and every input script, without exception.**

This is enforced in the writer via ASE's `specorder`, and every data file declares
both types even when one species is absent, so the same `pair_coeff` line works
for every candidate.

**Verify the species map once, before any science.** For `eam/alloy` and `pace`
alike, it is the element list on the `pair_coeff` line — *not* the order inside
the potential file — that sets the map. Run a single-point evaluation on a pure
cell of each element with the map, and again with the map deliberately swapped:
the energies must differ by roughly an eV per atom. An inverted species map is a
silent catastrophe that survives every other check.

### 4.3 The data tree

Code lives in this repository; **data lives outside it**, under `data_tree` from
the config, one directory per potential ("thread"):

```
<data_tree>/
├── 00_STRUCTURE-POOL/                      shared by all potentials
│   ├── fcc_Ni.lmps  fcc_Ni.yaml            structure + provenance sidecar
│   └── …
├── 01_HULL-<potential-A>/
│   ├── RELAX/<name>/                       inputs, logs, relaxed .lmps, run.yaml
│   ├── relax-summary_<pot>.yaml            one record per structure
│   ├── hull-table_<pot>_<system>.dat       ← THE interchange product
│   ├── hull-analysis_<pot>.yaml            machine-readable everything
│   ├── hull-plot_<pot>_<system>.pdf
│   ├── mu-vs-x_<pot>_<system>.dat / .pdf
│   ├── RESULTS_<pot>.md                    prose findings
│   ├── VERIFY-<name>/                      perturbed-restart records
│   └── STRAIN-001-Ni/                      phase 4 outputs (name from the epitaxy model)
└── 02_HULL-<potential-B>/                  same again
```

The structure pool is built **once** and reused by every potential, so all
potentials see byte-identical starting structures.

### 4.4 The phases

| phase | command(s) | gate |
|---|---|---|
| 0 environment audit | *(manual)* | binary + packages + species map verified |
| 1 structure pool | `build_pool`, `mp_pool` | pool + sidecars written |
| 2 relaxation | `relax` *or* `prepare`→`execute`→`harvest` | every candidate through all integrity gates |
| 3 hull + analysis | `hull`, `report` | pymatgen and scipy hulls agree |
| 4 strained hull | `strain`, `strain_hull` | all ε points converged |

Phases run in order. Each ends with its report written and its criteria stated
PASS/FAIL. `verify` can be invoked at any point after phase 2.

### 4.5 Two execution paths, one result

Phase 2 has two interchangeable implementations:

| | **integrated** (`relax`) | **decoupled** (`prepare` → `execute` → `harvest`) |
|---|---|---|
| processes | one: render, run and check in a loop | three separate commands |
| needs LAMMPS at | run time | `execute` only — `prepare` and `harvest` need none |
| needs the venv at | all times | `prepare`/`harvest` only — `execute` is stdlib-only |
| where compute happens | this machine | anywhere: laptop, workstation, SLURM array |
| output | `relax-summary_<pot>.yaml` | **the same file, same schema** |

The decoupled path renders *every* input up front — the box-relax stage, all
fallback cycles, and the equivalence single-point — into a self-contained `RELAX/`
tree, so the executing machine needs no template engine, no config and no venv.
The two paths are asserted **bit-identical** in the test suite and were verified
structure-by-structure during validation.

Use the integrated runner on a workstation. Use the decoupled path when the
compute belongs somewhere else — that is the whole point of [section 7](#7-running-on-an-hpc-cluster).

---

## 5. Manual — command reference

Every module is a `python -m` entry point taking the config path as its first
argument and (from phase 2 onward) a potential key as its second. Installing the
package with `pip install -e .` also provides a `hull0k` console script:
`hull0k relax configs/Ni-Cu.yaml EAM-Fischer2018` is equivalent to
`python -m hull0k.relax configs/Ni-Cu.yaml EAM-Fischer2018`.

Trailing structure names, where accepted, restrict the command to those
structures. Omit them to process the whole pool.

---

### `hull0k.build_pool` — phase 1: build the candidate structure pool

```bash
python -m hull0k.build_pool <config> [endmembers] [prototypes] [sqs] [defects]
```

Writes `<name>.lmps` + `<name>.yaml` into `<data_tree>/<pool_dir>/`. With no
sub-pool arguments it builds all four.

| sub-pool | what it builds |
|---|---|
| `endmembers` | fcc, bcc and hcp of **both** elements — the metastability anchors of the pure elements |
| `prototypes` | the binary suite, both orderings where the prototype is not exchange-symmetric |
| `sqs` | special quasirandom structures via `icet`, one per composition in `sqs.compositions` |
| `defects` | vacancies and substitutionals in supercells of both end members, plus a larger spot check and its perfect-host reference |

**Prototypes currently implemented** (`hull0k/prototypes.py`):

`A1` fcc · `A2` bcc · `A3` hcp · `L1_2` (A₃B) · `L1_0` (AB) · `B2` · `B1` ·
`B3` · `D0_22` (A₃B) · `D0_19` (A₃B) · `D0_a` (β-Cu₃Ti, A₃B) · `C11_b` (A₂B) ·
`C15` (Laves, A₂B) · `A15` (A₃B) · `L1_1` (CuPt) · `Z2` ([001] superlattice) ·
`40` ([201] superlattice)

Cubic and Laves prototypes are expanded from Wyckoff positions via
`ase.spacegroup.crystal`; the decorated-fcc superlattices (`L1_1`, `Z2`, `40`) are
built by plane decoration of an fcc supercell. Every structure's space group is
determined with spglib **as built** and recorded in the sidecar, so a mistyped
Wyckoff set shows up immediately rather than three phases later.

Initial lattice parameters are Vegard-interpolated from `a_fcc` in the config at
constant atomic volume. They are *guesses only* — phase 2 relaxes everything.

Each candidate is assigned a `box_coupling` class by its symmetry, which decides
the `fix box/relax` mode used in phase 2:

| symmetry | coupling | prototypes |
|---|---|---|
| cubic | `iso` | fcc, bcc, L1_2, B2, B1, B3, C15, A15 |
| tetragonal / orthorhombic | `aniso` | L1_0, D0_22, D0_a, SQS and defect supercells |
| hexagonal / trigonal / triclinic | `tri` | hcp, D0_19, L1_1, `40`, Materials Project entries |

> `aniso` on a hexagonal cell **silently breaks a = b**. When in doubt use `tri`.

**Options in the config:**

```yaml
sqs:
  parent: fcc                        # fcc | bcc | hcp — icet is lattice-agnostic
  compositions: [0.25, 0.50, 0.75]   # x_B
  natoms: 108
  cutoffs: [8.0, 5.0]                # pair, triplet cluster-space cutoffs (Å)
  n_steps: 50000
defects:
  host_lattice: fcc                  # fcc | bcc | hcp
  supercell: [4, 4, 4]               # → 256 sites for fcc
  spotcheck_supercell: [6, 6, 6]     # → 864 sites, one case, finite-size shift
  cases: [vac_Ni, vac_Cu, sub_CuinNi, sub_NiinCu]
```

The SQS sidecar records the cluster-space cutoffs, the number of Monte-Carlo
steps, and the **achieved maximum correlation mismatch** against the ideal random
cluster vector — read it before trusting an SQS.

---

### `hull0k.mp_pool` — phase 1c: Materials Project entries

```bash
python -m hull0k.mp_pool <config>
```

Queries `materials_project.chemsys` through `mp-api`, pulls every entry (stable
and above-hull), and deduplicates against the already-built pool with pymatgen's
`StructureMatcher`:

- **match** → no new candidate; the matching pool sidecar gains an `mp_matches`
  block with the MP id, the DFT formation energy and the MP URL.
- **no match** → written as a new candidate `mp_<id>_<formula>`.

The MP DFT formation energy becomes the `Ef_DFT_MP_eV` **comparison column** in
the hull table and the faint grey markers in the plot. It is *never* mixed into
the hull — the hull is built from potential energies only.

**API key.** The key is read from the environment variable
`MATERIALS_PROJECT_API_KEY`. If it is not in the environment — the usual case,
since rc-file exports never reach a non-interactive process — an interactive
login shell is asked instead: your own `$SHELL` first, then `zsh`, then `bash`,
so no particular shell is assumed. The key is passed **explicitly** to
`MPRester(api_key=…)` — note that `mp-api` otherwise looks for `MP_API_KEY` /
`PMG_MAPI_KEY`. It is never printed and never written to any file in this
repository or the data tree.

This step is optional. Skip it and every `Ef_DFT_MP_eV` field reads `NA`.

---

### `hull0k.relax` — phase 2, integrated runner

```bash
python -m hull0k.relax <config> <potential-key> [--maxiter=N] [names...]
```

For each structure: render → run → check, in one process.

1. **Stage 1** — one-shot `minimize-boxrelax.in.skel`: `min_style cg`,
   `min_modify line quadratic`, `fix box/relax <coupling> 0.0`.
2. **Two-stage fallback**, entered when stage 1 stalls or the pressure gate fails:
   alternate fixed-cell minimisation and box-relax, up to 6 cycles, until both
   converge. The path taken is recorded per structure (`boxrelax` /
   `twostage-fallback` / `…-UNCONVERGED`).
3. **Integrity checks** — symmetry before/after, `d_min`, per-component pressure.
   The pressure gate tests only the components the coupling actually relaxes;
   unrelaxed shear on `aniso` cells is recorded as a diagnostic rather than
   failing the run.
4. **Equivalence gate** — an independent `run 0` on the written relaxed file.

`--maxiter=N` overrides the iteration cap (default `max(2000, 20·N_atoms)`) — use
a small value for a smoke test. Writes `RELAX/<name>/run.yaml` per structure and
merges one record per structure into `relax-summary_<pot>.yaml`; re-running a
subset updates only those records.

---

### `hull0k.prepare` / `hull0k.execute` / `hull0k.harvest` — phase 2, decoupled

```bash
python -m hull0k.prepare <config> <potential-key> [names...]     # no LAMMPS needed
python -m hull0k.execute <config> <potential-key> [--jobs=N] [names...]
python -m hull0k.harvest <config> <potential-key> [names...]     # needs the venv
```

**`prepare`** renders every input for every structure and makes
`<thread>/RELAX/` self-contained for execution anywhere:

```
RELAX/
├── manifest.txt              one structure name per line — array index = line number
├── run_one.py                stdlib-only driver (no venv required)
├── run-all-local.sh          serial bash loop over the manifest
├── submit-hull0k-array.sh    SLURM array TEMPLATE with EDITME_ tokens
├── README-EXECUTE.md         the execution steps, generated with the tree
└── <name>/
    ├── <name>.lmps                          the starting structure
    ├── <label>-min.in                       stage 1
    ├── <label>-c{0..5}a-fixmin.in           fallback, fixed-cell half
    ├── <label>-c{0..5}b-min.in              fallback, box-relax half
    ├── <label>-equiv-singlepoint.in         the equivalence gate
    └── job.json                             file names, gates, the full lmp command
```

**`execute`** runs `run_one` over the manifest locally. `--jobs=N` runs N
structures concurrently (independent single-core LAMMPS processes; pick N at or
below your core count).

**`harvest`** walks the executed tree — whether it ran here or came back from a
cluster — applies every venv-dependent check (spglib symmetry, `d_min`, the
pressure gate, the equivalence comparison), writes `run.yaml` per structure, and
merges `relax-summary_<pot>.yaml` **into the thread directory next to the
results**. Structures not yet executed are recorded as `NOT-EXECUTED` rather than
silently skipped.

The summary schema is identical to the integrated runner's, so `hull` and
`report` consume either without knowing where the compute happened. You may even
mix: relax some structures locally and others on a cluster.

---

### `hull0k.verify` — perturbed-restart verification

```bash
python -m hull0k.verify <config> <potential-key> <name> [nseeds]
```

Re-relaxes one candidate from `nseeds` perturbed starting points (positions
rattled by a 0.05 Å Gaussian, cell isotropically scaled by ±1 %) and compares each
final energy per atom against the unperturbed result.

Use it on **any surprising entry** — anything that lands on or below the hull
where you did not expect it. Results go to `VERIFY-<name>/verify-<name>.yaml` and
are deliberately **not** merged into `relax-summary`, so the hull table stays a
single clean record per structure. `report` picks the verification files up
automatically and quotes them in `RESULTS_<pot>.md`.

If perturbed restarts reconverge to the same minimum, a negative formation energy
is a **property of the potential**, not a relaxation artefact — report it
prominently rather than tuning it away.

---

### `hull0k.hull` — phase 3: hull, chemical potentials, defect energies

```bash
python -m hull0k.hull <config> <potential-key>
```

Reads the pool sidecars + `relax-summary_<pot>.yaml`, re-reads every relaxed
`.lmps` for its final cell parameters, and writes:

- `hull-table_<pot>_<system>.dat` — the interchange product ([schema](#61-hull-table_pot_systemdat))
- `hull-plot_<pot>_<system>.pdf` — two panels: full range, plus a zoom on
  `E_f < 0.16 eV/atom` with the number of clipped entries and the maximum `E_f`
  stated in the panel title. Converged candidates filled, not-assured candidates
  open, DFT comparison as faint grey squares.
- `mu-vs-x_<pot>_<system>.dat` / `.pdf` — the tangent construction per hull segment
- `hull-analysis_<pot>.yaml` — machine-readable: hull members from both methods,
  the cross-check verdict, segments, defect energies

Printed to stdout: hull members from pymatgen, from scipy, whether they agree,
and every defect energy.

**Defect energies** use the relaxed pure fcc end members of *this potential* as
reservoirs (a constant-μ convention, stated in every output file):

```
E_vac(host)          = E_cell − N·E_coh(host)
E_sub(solute@host)   = E_cell − (N−1)·E_coh(host) − E_coh(solute)
```

The dilute hull slope, `N · E_f` of the defect cell, is reported alongside — it
equals the substitution energy by construction, and the SQS chord slopes must
extrapolate to the same value. That agreement is the built-in consistency check
between the defect branch and the solution branch; the residual difference is the
finite-`x` curvature, quantified in the report.

---

### `hull0k.report` — phase 3: prose findings

```bash
python -m hull0k.report <config> <potential-key>
```

Composes `RESULTS_<pot>.md` in the thread directory: validation gates against the
config's `references` block, hull membership (with any verification records
quoted), the full `e_above_hull` spectrum as a table, chemical potentials per
segment, the defect-energy table with its reservoir convention, the dilute-limit
consistency check, and — once a second potential has been analysed — a
potential-vs-potential-vs-DFT comparison table.

Run `hull` for **all** potentials before running `report`, so the comparison
tables can be filled in.

---

### `hull0k.strain` — phase 4: epitaxial relaxation driver

```bash
python -m hull0k.strain <config> <potential-key> \
    [--eps=-0.04,-0.03,...] [--include-sqs] [--include-defects] [names...]
```

For each eligible candidate and each strain ε: take the **relaxed bulk** structure
from phase 2, hand it to the configured epitaxy model, write the strained cell as
a new `.lmps` with its own sidecar recording every choice made, and relax it with
`minimize-epitaxial.in.skel` (`fix box/relax z 0.0` — only `lz` and the internal
coordinates move).

- default grid: ε = −4 % … +4 % in 1 % steps.
- `--eps=` takes an explicit comma-separated list; re-runs **merge** into the
  existing per-structure summary, so a refinement grid extends an earlier sweep
  rather than replacing it.
- SQS cells run only with `--include-sqs`, defect cells only with
  `--include-defects` (they multiply the cost, though these runs are cheap in
  absolute terms).
- Candidates the model cannot orient are skipped, each with a written reason, in
  `<strain-dir>/skipped-candidates.yaml`. **Read that file** — it is where the
  physics choices are recorded.

Outputs go to `<thread>/<strain-dir>/`, named by the selected epitaxy model:
the model's `dir_tag` with the config's elements substituted, e.g.
`STRAIN-001-Ni` for the fcc(001) model on a Ni substrate. Two orientations
therefore never overwrite each other. `epitaxy: {strain_dir: …}` pins the name
explicitly if you want one.

Which candidates are eligible, what the substrate reference is, and how the film
cell is constructed all live in the epitaxy model, not in this driver
(see [section 9.4](#94-a-new-epitaxy-orientation-or-substrate)).

---

### `hull0k.strain_hull` — phase 4: strained-hull analysis

```bash
python -m hull0k.strain_hull <config> <potential-key>
```

Builds a hull per ε point in **both** reference conventions, asserts that
`e_above_hull` agrees between them, and writes:

- `hull-table-strained_<pot>_<system>_<epstag>.dat` — one per ε ([schema](#62-hull-table-strained_pot_system_epstagdat))
- `strain-trajectories_<pot>_<system>.dat` — long format, every structure × every ε
- `defect-energies-vs-eps_<pot>_<system>.dat` — defect energies against **coherent**
  reservoirs (end members strained the same way at the same ε)
- `strain-stability_<pot>_<system>.pdf` — `e_above_hull(ε)` trajectories plus an
  `(x, ε)` stability map: filled = on the coherent hull, open = above it
- `RESULTS-STRAIN_<pot>.md` — prose, including a **crossings** section listing
  every ε interval in which a structure enters or leaves the hull

A crossing is the headline result of phase 4: a structure that strain stabilises
or destabilises on this substrate.

---

## 6. Output formats (the interchange contract)

**File format rules, applied everywhere:** plain text only — Markdown, YAML, and
space-separated `.dat` (never CSV). Every `.dat` file that is not native LAMMPS
format carries `#` header lines naming every column with its units, plus the
generating date, the potential and its SHA-256. Files ending `.lmps` are LAMMPS
data files; files ending `.dump` are dumps. Directories are CAPS; element symbols
keep canonical case (`Ni-Cu`, not `NI-CU`).

### 6.1 `hull-table_<pot>_<system>.dat`

**This schema is a contract** — downstream consumers and companion projects depend
on it. Extend it at the right-hand end; never reorder or rename.

```
name prototype source x natoms E_per_atom_eV Ef_per_atom_eV e_above_hull_eV
     a_A b_A c_A alpha_deg beta_deg gamma_deg spacegroup_relaxed drifted(0/1)
     Ef_DFT_MP_eV(or NA)
```

| column | meaning |
|---|---|
| `name` | pool entry name; the key everywhere else |
| `prototype` | structure-type label as built (spaces → `_`) |
| `source` | first token of the provenance string (`hull0k.prototypes`, `icet`, `Materials`, …) |
| `x` | `N_B/(N_A+N_B)`, atoms only |
| `natoms` | atoms in the relaxed cell |
| `E_per_atom_eV` | LAMMPS potential energy per atom, absolute |
| `Ef_per_atom_eV` | formation energy vs this potential's relaxed fcc end members |
| `e_above_hull_eV` | distance to the lower convex hull |
| `a_A … gamma_deg` | relaxed cell parameters |
| `spacegroup_relaxed` | spglib symbol + number after relaxation, whitespace stripped |
| `drifted` | 1 if the space group changed during relaxation |
| `Ef_DFT_MP_eV` | Materials Project DFT comparison, or `NA` |

### 6.2 `hull-table-strained_<pot>_<system>_<epstag>.dat`

The bulk contract plus `eps`, with **both** formation-energy conventions:

```
name prototype x natoms eps E_per_atom_eV Ef_unstrained_ref_eV Ef_coherent_ref_eV
     e_above_hull_eV
```

The ε tag encodes sign and value without a decimal point: `epsp0025` = +2.5 %,
`epsm0040` = −4.0 %.

### 6.3 `relax-summary_<pot>.yaml`

One record per structure — the phase-2 → phase-3 handover. Written identically by
the integrated and decoupled paths:

```yaml
- name: S40_NiCu
  potential: EAM-Fischer2018
  status: OK                       # OK | FAILED | NOT-EXECUTED
  path: boxrelax                   # boxrelax | twostage-fallback | …-UNCONVERGED
  natoms: 8
  E_total_eV: -31.96154…
  E_per_atom_eV: -3.99519…
  fnorm_per_dof_eV_A: 4.7e-12      # the reported convergence measure
  fnorm_per_atom_eV_A: 2.3e-11
  press_components_bar: {press: …, pxx: …, …, pyz: …}
  press_gate_components: [press, pxx, pyy, pzz, pxy, pxz, pyz]   # by coupling
  unrelaxed_shear_residual_bar: {}                               # diagnostic only
  press_ok: true
  min_dist_A: 2.4884
  min_dist_ok: true
  spacegroup_before: I4_1/amd (141)
  spacegroup_after:  I4_1/amd (141)
  symmetry_kept: true
  cell_relaxed: {lx: …, ly: …, lz: …, xy: …, xz: …, yz: …}
  equiv_gate_eV_per_atom: 0.0      # < 1e-8 required
  equiv_ok: true
  relaxed_file: S40_NiCu_EAM-Fischer2018-relaxed.lmps
  coupling: tri
  date: '2026-08-26'
```

### 6.4 Structure sidecar `<name>.yaml`

Written next to every `<name>.lmps`: prototype label, provenance source, composition,
`natoms`, `x_<B>`, space group **as built**, cell parameters, the assigned
`box_coupling`, and the date — plus sub-pool extras (`defect`, `host`, `solute`,
`parent_sites`; `sqs_cutoffs_A`, `sqs_correlation_mismatch_max`; `mp_id`,
`mp_url`, `mp_formation_energy_per_atom_eV`, or an `mp_matches` list on a merged
duplicate).

### 6.5 Per-run provenance `RELAX/<name>/run.yaml`

Potential key, file path and SHA-256; the LAMMPS command and version; the template
version; the full result record. This is what makes any published number traceable
back to a specific binary and a specific potential file.

---

## 7. Running on an HPC cluster

`hull0k` **never logs into a cluster.** It does not ssh, submit, poll or pull.
It produces a tree that is trivially stageable and a submission script that is one
edit away from working, and leaves the human in control of the queue. That is a
deliberate design decision, not a missing feature: batch systems, module stacks
and scratch policies differ too much to automate blindly, and a workflow that
submits jobs on its own is a workflow you cannot review before it spends your
allocation.

### 7.1 When you need a cluster

For a Ni–Cu-sized pool (40 structures of 2–864 atoms with an EAM potential) you do
not: the whole phase-2 batch takes minutes on a laptop. You need a cluster when

- the potential is expensive — ACE, GRACE, MACE, SNAP and friends are orders of
  magnitude slower than EAM,
- the cells are large — big SQS cells, big defect supercells, long-period
  superlattices,
- the pool is large — a wide prototype sweep, a fine ε grid, several potentials,
- or the potential is only *built* on the cluster (a Kokkos/GPU or MPI build you
  cannot reproduce locally).

Phase 2 and phase 4 are the expensive phases and both are embarrassingly parallel:
one structure per array task, no communication.

### 7.2 The model

```
  workstation                       cluster                      workstation
 ┌─────────────┐   rsync   ┌────────────────────────┐   rsync   ┌────────────┐
 │  build_pool │  ───────► │  sbatch array          │  ───────► │  harvest   │
 │  prepare    │           │  → run_one.py per task │           │  hull      │
 └─────────────┘           └────────────────────────┘           │  report    │
   venv needed              plain python3 + lmp only            └────────────┘
                                                                  venv needed
```

`run_one.py` is **stdlib-only by design**: it reads `job.json`, calls the `lmp`
command named there, parses the logs with `re`, and writes `result-status.json`.
No numpy, no ASE, no config file, no virtual environment on the compute node.
Every venv-dependent check happens later, in `harvest`, back on your workstation.

### 7.3 Step by step

**1 · Point the config at the cluster — before preparing.**

This is the step that catches people out. `prepare` **bakes** the `pair_coeff` line
and the `lmp` command into the rendered inputs and into `job.json`. They must
already be the cluster's values:

```yaml
lmp_binary: /cmmc/ptmp/<user>/BIN/lmp            # the CLUSTER path
lmp_launch: "srun"                               # MPI-built module; "" for a serial binary
potentials:
  ACE-CuNi:
    file: /cmmc/ptmp/<user>/POTENTIALS/ACE_CuNi.yaml   # the CLUSTER path
```

`data_tree` stays a **local** path: the tree is built here, rsynced there, and
rsynced back, and `harvest` reads it here. Only `lmp_binary`, `lmp_launch` and the
potential `file` paths change. (The cluster potential path then also appears in
`run.yaml` — which is correct provenance: that *is* the file the energies came from.)

**2 · Build the pool and prepare, locally.**

```bash
python -m hull0k.build_pool configs/<system>.yaml
python -m hull0k.prepare    configs/<system>.yaml <potential-key>
```

Neither step runs LAMMPS.

**3 · Stage to cluster scratch — never `$HOME`.**

```bash
rsync -av --exclude '*.log' \
  <data_tree>/<thread_dir>/RELAX/ \
  <user>@<cluster>:/scratch/<user>/<project>/RELAX/
```

Copy the potential files to exactly the paths `job.json` names. Cluster `$HOME`
quotas are tight and unforgiving; scratch is where compute belongs. Scratch is
also usually **not backed up** — everything you care about comes back to the
workstation before you clean up.

**4 · Fill in the `EDITME_` tokens.**

`prepare` writes `submit-hull0k-array.sh` as a template. Every cluster-specific
value is a token that must be replaced before submission:

| token | what to put there |
|---|---|
| `EDITME_PARTITION` | the queue for small single-core jobs (shared, not exclusive) |
| `EDITME_WALLTIME` | generous per-task wall time; unused time is normally not billed |
| `EDITME_MODULE_LOAD_LINE` | the `module load …` line(s) after `module purge` |

The array is sized to the manifest: array index *n* runs manifest line *n*.

**5 · Probe with one task, then release the rest.**

```bash
sbatch --test-only submit-hull0k-array.sh    # validates placement, queues nothing
sbatch --array=1-1 submit-hull0k-array.sh    # ONE task
```

Read that task's output and its `result-status.json` before submitting the rest.
Never let a full array be the first thing you learn from.

```bash
sbatch --array=2-40%20 submit-hull0k-array.sh   # the rest, max 20 concurrent
```

The `%N` throttle is worth using on a large pool: these tasks are seconds long,
and a thousand-task array that starts all at once is unkind to a shared filesystem
and to your fair-share.

**6 · Pull back and harvest.**

```bash
rsync -av <user>@<cluster>:/scratch/<user>/<project>/RELAX/ \
          <data_tree>/<thread_dir>/RELAX/
python -m hull0k.harvest configs/<system>.yaml <potential-key>
```

`harvest` applies every integrity gate and writes `relax-summary_<pot>.yaml` into
the thread directory next to the results. From here on, phases 3 and 4 do not care
where the compute happened.

Pull back the `.in`, `.lmps`, `.json`, `.yaml` and `.dat` files; the per-iteration
LAMMPS `.log` files and the Slurm `.out`/`.err` files are re-generatable noise
unless you are debugging.

### 7.4 A worked cluster example (SLURM, MPI-built LAMMPS module)

`submit-hull0k-array.sh` after editing, for a machine with a shared sub-node
partition and a site LAMMPS module:

```bash
#!/bin/bash
#SBATCH --job-name=hull0k-relax
#SBATCH --partition=s.cmmg              # ← EDITME_PARTITION: shared, sub-node
#SBATCH --time=00:30:00                 # ← EDITME_WALLTIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-40
#SBATCH -o hull0k-relax-%A_%a.out
set -euo pipefail
module purge
module load lammps/250722               # ← EDITME_MODULE_LOAD_LINE

cd "$SLURM_SUBMIT_DIR"
name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
[[ -n "$name" ]] || { echo "no manifest entry for task $SLURM_ARRAY_TASK_ID"; exit 1; }
[[ -f "$name/job.json" ]] || { echo "missing $name/job.json"; exit 1; }
python3 run_one.py "$name"
```

with, in the config used for `prepare`:

```yaml
lmp_binary: lmp          # provided by the module on PATH
lmp_launch: "srun"       # the module's binary is MPI-linked
```

### 7.5 Cluster gotchas

- **Never run `lmp` on a login node.** MPI-linked builds call `MPI_Init()` before
  processing any flag — including `-h` — and abort. Every invocation, diagnostic
  probes included, goes through the batch system.
- **`lmp_launch: "srun"` is required for MPI-built binaries**, even for a
  single-task job. A serial build needs an empty `lmp_launch`.
- **`module purge` then explicit `module load`.** Do not rely on inheriting the
  submitting shell's environment; some Slurm builds reject `--get-user-env`
  outright, and a job whose environment depends on your login shell is a job that
  breaks the day you change your dotfiles.
- **Never install into cluster `$HOME`.** `run_one.py` exists precisely so you do
  not have to build a virtual environment on the cluster.
- **A network filesystem is not a local disk.** Each array task writes several
  small files into its own directory; that is fine. Throttle wide arrays with `%N`.
- **`data_tree` in a cluster-facing config is still the local path.** Only the
  binary, the launcher and the potential paths point at the cluster.
- **Check `result-status.json`, not just the exit code.** A task can exit 0 having
  written `status: FAILED`, and `harvest` will tell you — but only after you have
  pulled the tree back.

### 7.6 Other batch systems

Only a SLURM array template ships. For PBS/Torque, LSF or SGE, the adaptation is
a few lines and the interface is stable: something must set an array index, read
the *n*-th line of `manifest.txt`, and call `python3 run_one.py <name>`. Everything
else — the inputs, the gates, the fallback logic — is already in the tree.

---

## 8. Walked-through example: Ni–Cu with an EAM potential

This is the validation system, reproduced end to end. **Ni–Cu is the right
validation case** precisely because the answer is known: the assessed phase diagram
has *no* stable intermetallics, so the correct 0 K hull is the two fcc end members
and every ordered compound and solid solution should sit at positive formation
energy. Anything else is either a bug or a statement about the potential — and
`hull0k` is built to tell those apart.

All numbers below are the actual output of the run.

### Step 0 — write the config

`configs/Ni-Cu.yaml`, abridged:

```yaml
system: Ni-Cu
elements: [Ni, Cu]                 # type 1 = Ni, type 2 = Cu — fixed globally
masses: [58.6934, 63.546]
a_fcc: {Ni: 3.52, Cu: 3.615}       # building guesses only

data_tree: /path/to/SIMULATIONS/CONVEX-HULL-0K-Ni-Cu
pool_dir: 00_STRUCTURE-POOL

potentials:
  EAM-Fischer2018:
    thread_dir: 01_HULL-EAM-Fischer2018
    file: /path/to/Cu_Ni_Fischer_2018.eam.alloy
    sha256: ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240
    pair_style: eam/alloy
    pair_coeff: "* * POTPATH Ni Cu"          # POTPATH is substituted with `file`
    citation: "F. Fischer, G. Schmitz, S. M. Eich, Acta Mater. 176, 220 (2019)"

lmp_binary: /path/to/lmp-pace
lmp_launch: ""                     # "srun" on an MPI cluster

sqs:     {parent: fcc, compositions: [0.25, 0.50, 0.75], natoms: 108,
          cutoffs: [8.0, 5.0], n_steps: 50000}
defects: {host_lattice: fcc, supercell: [4,4,4], spotcheck_supercell: [6,6,6]}
epitaxy: {model: fcc001-cube-on-cube}
materials_project: {chemsys: Cu-Ni}

references:                        # optional: phase-3 validation gates
  ACE-CuNi: {a0_Ni: 3.52196786648779, Ecoh_Ni: -4.81899412302984, …}
```

> **Note the element order on the `pair_coeff` line.** The Fischer file's *internal*
> order is `Cu Ni`, but for `eam/alloy` it is the `pair_coeff` element list that
> sets the type map — so `* * <file> Ni Cu` correctly gives type 1 = Ni. Verify it
> with a single-atom check before trusting anything downstream.

### Step 1 — build the structure pool

```bash
python -m hull0k.build_pool configs/Ni-Cu.yaml
python -m hull0k.mp_pool    configs/Ni-Cu.yaml     # optional: the DFT column
```

40 candidates, built once and shared by both potentials:

| sub-pool | count | entries |
|---|---|---|
| end members | 6 | fcc/bcc/hcp × {Ni, Cu} |
| binary prototypes | 21 | L1_2, L1_0, B2, B1, B3, D0_22, D0_19, D0_a, C11_b, C15, A15 (both orderings where applicable), L1_1, Z2, `40` |
| Materials Project | 4 | new entries; 2 further MP entries matched existing prototypes and were merged as provenance instead |
| SQS | 3 | fcc, 108 atoms, x_Cu = 0.25 / 0.50 / 0.75 |
| defects + references | 6 | vacancy in Ni and in Cu, Cu-in-Ni and Ni-in-Cu (256 sites), the 864-site spot check, and its perfect-host reference |

### Step 2 — relax everything

```bash
python -m hull0k.relax configs/Ni-Cu.yaml EAM-Fischer2018
```

40/40 converged. The two-stage fallback was needed for three structures (B3, C11_b,
D0_a) and the path taken is recorded for each. The **equivalence gate returned
exactly 0.0** for all 40 runs — and for all 80 across both potentials. No structure
changed space group during relaxation.

To do the same work on a cluster, replace this one command with the decoupled path
of [section 7](#7-running-on-an-hpc-cluster); the resulting
`relax-summary_EAM-Fischer2018.yaml` is identical.

### Step 3 — build the hull

```bash
python -m hull0k.hull   configs/Ni-Cu.yaml EAM-Fischer2018
python -m hull0k.report configs/Ni-Cu.yaml EAM-Fischer2018
```

```
hull members (pymatgen): ['S40_NiCu', 'fcc_Cu', 'fcc_Ni']
hull members (scipy):    ['S40_NiCu', 'fcc_Cu', 'fcc_Ni']
cross-check agree: True
  Evac_Ni_256sites = 1.570985
  Evac_Cu_256sites = 1.271739
  Esub_CuinNi_256sites = 0.146867
  Esub_NiinCu_256sites = 0.093489
```

**This is a finding, not a bug.** The expected hull was `{fcc Ni, fcc Cu}`, and a
third member appeared: the `40` (CuPt-type [201]) superstructure at
x = 0.5, which this EAM potential places at

```
S40_NiCu  40  hull0k.prototypes  0.500000  8  -4.00100957  -0.00600960  0.00000000
          3.740758 3.740758 6.640798  90 90 90  I4_1/amd(141)  0  NA
```

— `E_f = −6.0 meV/atom`, i.e. **below zero and on the hull**. The prescribed
response is to verify, not to fix:

```bash
python -m hull0k.verify configs/Ni-Cu.yaml EAM-Fischer2018 S40_NiCu 3
```

Three perturbed restarts (rattle 0.05 Å, cell ±1 %) reconverge **bit-identically**
to the same `I4_1/amd` minimum. The negative formation energy is a property of
the Fischer 2018 parameterisation, not a relaxation artefact, and it is reported
prominently in `RESULTS_EAM-Fischer2018.md`. `report` picks the verification record
up automatically.

The hull's chemical potentials show the same story — two segments instead of one,
kinked at the `40` phase:

```
# x_lo   x_hi   mu_Ni_eV     mu_Cu_eV     dmu_eV      segment_between
0.000000 0.500000 -4.44999999 -3.55201916 0.89798082 fcc_Ni--S40_NiCu
0.500000 1.000000 -4.46201918 -3.53999997 0.92201921 S40_NiCu--fcc_Cu
```

**The dilute-limit consistency check.** The defect branch and the solution branch
must agree in the dilute limit, and they do, with exactly the expected finite-`x`
curvature between them:

| quantity | value (eV) |
|---|---|
| `Esub(Cu in Ni)`, 256 sites | 0.146867 |
| `Esub(Cu in Ni)`, 864 sites | 0.146897 (finite-size shift: **+0.03 meV**) |
| SQS chord slope, x → 0.25 | 0.097065 |
| SQS chord slope, x → 0.50 | 0.056322 |

The chord slopes fall away from the dilute value monotonically as `x` grows —
curvature of `E_f(x)`, exactly as it should be. The 256 → 864 site shift of
0.03 meV says the 256-site cell is already converged for this quantity.

### Step 4 — the strained (epitaxial) hull

```bash
python -m hull0k.strain      configs/Ni-Cu.yaml EAM-Fischer2018 --include-sqs --include-defects
python -m hull0k.strain_hull configs/Ni-Cu.yaml EAM-Fischer2018
```

Substrate: fcc Ni (001), cube-on-cube, a₀_ref = this potential's own relaxed Ni
lattice constant. ε from −4 % to +4 % in 1 % steps, refined to 0.5 % between
+1.5 % and +3.5 % where the interesting behaviour showed up. 354/354 epitaxial
relaxations converged.

B1 and B3 were **dropped with a written reason**: they are open structures with no
1:1 cube-on-cube commensurability with the fcc substrate, and forcing `lx` to
a₀_ref would mean about −20 % strain. hcp, D0_19, L1_1, C11_b, C15, A15, D0_a, bcc
and B2 are likewise skipped by the model's eligibility list. All of it is in
`STRAIN-001-Ni/skipped-candidates.yaml` — the file exists so that the choices are
auditable rather than invisible. (That directory name is not hard-coded: it is the
fcc(001) model's `dir_tag` with this config's substrate element substituted.)

The headline:

```
**FINDING: strain changes hull membership.**
- between eps = +0.025 and +0.030: `S40_NiCu` enters the hull.
```

Read physically: the Ni-substrate constraint at ε = 0 pushes the (bulk-stable)
`40` phase **off** the coherent hull — below +2.5 % the coherent hull is
`{strained fcc Ni, strained fcc Cu}` everywhere — and it returns once the in-plane
parameter approaches Cu's lattice constant. No defect or SQS cell touches the EAM
hull at any ε.

Running the same phase 4 with the ACE potential gives a qualitatively different
picture — dilute Ni-in-Cu is exothermic for ACE, and that pocket survives the
epitaxial constraint at *every* ε on the grid — which is the point of running two
potentials through an identical pipeline.

### What you end up with

```
01_HULL-EAM-Fischer2018/
├── RELAX/…                                     40 run directories, full provenance
├── relax-summary_EAM-Fischer2018.yaml
├── hull-table_EAM-Fischer2018_Ni-Cu.dat        ← the interchange product
├── hull-analysis_EAM-Fischer2018.yaml
├── hull-plot_EAM-Fischer2018_Ni-Cu.pdf
├── mu-vs-x_EAM-Fischer2018_Ni-Cu.dat / .pdf
├── RESULTS_EAM-Fischer2018.md
├── VERIFY-S40_NiCu/verify-S40_NiCu.yaml
└── STRAIN-001-Ni/
    ├── hull-table-strained_…_eps{m,p}NNNN.dat  one per ε
    ├── strain-trajectories_…dat
    ├── defect-energies-vs-eps_…dat
    ├── strain-stability_…pdf
    ├── skipped-candidates.yaml
    └── RESULTS-STRAIN_EAM-Fischer2018.md
```

Repeat from step 2 with a second potential key; the pool, and therefore the
starting structures, are shared byte-for-byte.

---

## 9. Extending hull0k

### 9.1 A new binary system

Copy `configs/EXAMPLE.yaml`, set `system`, `elements`, `masses`, the `a_fcc`
guesses, the potential block(s), `data_tree`, `sqs.parent` and
`defects.host_lattice`. **No code changes.** Then:

```bash
python -m hull0k.build_pool configs/<new>.yaml
python -m hull0k.relax      configs/<new>.yaml <potential-key>
python -m hull0k.hull       configs/<new>.yaml <potential-key>
```

The `references:` block is optional — supply on-record `a0`, `E_coh`, `E_vac` or
`E_sub` values and `report` turns them into explicit PASS/FAIL validation gates.

### 9.2 A new potential for an existing system

Add a block under `potentials:` with its own `thread_dir`, file path, SHA-256,
`pair_style`, `pair_coeff` (using `POTPATH` as the file placeholder) and citation.
Re-run phases 2–4 with the new key. The pool is untouched, so the comparison is
exact. Make sure your `lmp` binary has the pair style's package built in.

### 9.3 A new prototype

Add a builder to `hull0k/prototypes.py` returning an `ase.Atoms` with `.info`
carrying `prototype` and `box_coupling`, then list it in `build_prototypes()`.
Two idioms are available: explicit fractional coordinates via `_mk()`, or Wyckoff
expansion via `ase.spacegroup.crystal` (see `C15`, `D0_19`, `D0_a`).

Get the `box_coupling` right — see the table in [section 5](#hull0kbuild_pool--phase-1-build-the-candidate-structure-pool)
— and check the sidecar's `spacegroup_as_built` against what you intended before
running anything.

### 9.4 A new epitaxy orientation or substrate

The strained-hull constraint layer is a registry. All orientation-specific logic
lives in `hull0k/epitaxy.py`; `strain.py`, `strain_hull.py` and the LAMMPS template
are orientation-agnostic — they only pin `lx`, `ly` and relax `z` plus internals on
whatever cell the model returns.

A model is four things:

```python
def _myref(cfg, potkey, thread, elements):
    """Resolve the substrate reference from the phase-2 results."""
    return {"a0_ref": …, "substrate": "…"}

def _myfilm(at0, side, eps, ctx):
    """Take the RELAXED bulk cell, apply the in-plane constraint at eps,
    return (strained Atoms, record dict). Record every choice made."""
    return at, {"eps": eps, "orientation": "… (choice, not optimum)", …}

EPITAXY_MODELS["fcc111-my-orientation"] = {
    "ref": _myref, "film": _myfilm,
    "eligible":     {"A1_fcc", "L1_2", …},   # prototypes it can orient
    "eligible_sqs": {"SQS_fcc"},             # gated behind --include-sqs
    "description":  "one line, written into every sidecar",
}
```

Select it with `epitaxy: {model: fcc111-my-orientation}`. Natural candidates:
fcc(111), a Bain-type bcc-on-fcc relationship, an hcp substrate, or a
lattice-matching search that finds commensurate supercells instead of assuming a
1:1 cube-on-cube fit.

The shipped model, `fcc001-cube-on-cube`, infers the number of conventional cells
in-plane from the atomic volume — valid for close-packed fcc-derived cells, which
is why its eligibility list is explicit and open structures are excluded.

### 9.5 A new parent lattice for SQS or defects

`sqs.parent` and `defects.host_lattice` accept `fcc`, `bcc` and `hcp` today
(`icet` itself is lattice-agnostic). To add another, extend `parent_lattice()` in
`hull0k/build_pool.py` — one function, and both sub-pools generalise together.
The bcc path is smoke-tested (54-atom SQS and bcc defect cells with correct spglib
symmetries).

---

## 10. Not yet implemented / known limitations

Written honestly, because knowing where a tool stops is more useful than a
feature list. Nothing here is broken; these are boundaries of the current scope.

**What is *not* on this list any more: system-specific behaviour.** The analysis
layer classifies every entry by its **prototype label** and by what the pool
builder recorded in its sidecar — never by the entry's name, never by the
elements. The strained-hull output directory is named by the epitaxy model, not
by a literal. LAMMPS version and binary come from the config. Defect hosts and
solutes come from the sidecar rather than from a composition guess. SQS
consistency checks fire on any parent lattice.
`tests/test_system_agnostic.py` enforces all of this against a bcc-hosted Ni–Nb
fixture, so a change that reintroduces a Ni–Cu-ism fails the test rather than
producing quietly wrong output on the next system.

### Physics

- **Binary systems only.** Formation energies, the hull, the tangent construction
  and the defect analysis all assume exactly two elements. A ternary needs a 2-D
  hull, a Gibbs-triangle plotter and a different `μ` construction. `pymatgen`'s
  `PhaseDiagram` would handle it; the surrounding code would not.
- **0 K static energies only.** No vibrational free energy, no quasi-harmonic
  approximation, no configurational entropy, no finite-temperature hull. The SQS
  cells give the *energy* of a disordered state, not its free energy — so nothing
  here can predict a miscibility gap or an order–disorder temperature on its own.
- **Zero pressure only.** `fix box/relax` is always called at 0.0; there is no
  pressure axis and no `E_f(x, P)`.
- **No zero-point energy correction.**
- **No magnetism, no charge.** Inherent to classical and most ML potentials, but
  worth stating when comparing against the DFT column: Materials Project entries
  carry magnetic configurations the potential knows nothing about.
- **Defects only in the pure end members.** Vacancies and substitutionals in
  ordered compounds and in SQS cells are deliberately out of scope. No
  interstitials, no defect clusters, no charged defects, and no finite-size
  correction scheme beyond the single larger spot-check cell.
- **Drift detection is coarse.** A candidate is flagged `drifted` when its spglib
  symbol changes. A relaxation that collapses into a *different* structure with
  the *same* space group is not caught, and the `StructureMatcher`-based
  identification of the drift product is not implemented — the flag tells you to
  look, not what it became.
- **SQS: one cell per composition.** No ensemble over several SQS realisations,
  and the achieved correlation mismatch is recorded but not gated against a
  target.

### Epitaxy / phase 4

- **One epitaxy model is implemented**: `fcc001-cube-on-cube`. The *registry* is
  general and the driver, the analysis and the template are orientation-agnostic —
  a second model is a registry entry, not a code change
  ([section 9.4](#94-a-new-epitaxy-orientation-or-substrate)) — but fcc(111),
  Bain-type bcc-on-fcc and hcp substrates are not written yet.
- **No orientation optimisation.** The orientation relationship is a *recorded
  choice*, not a search result. This is stated in every sidecar and every report,
  and it is the single largest open physics question in phase 4.
- **No lattice-matching / commensuration search.** The shipped model assumes a
  1:1 cube-on-cube fit and excludes structures that would need a supercell match;
  those are skipped with a written reason rather than approximated.
- **Only biaxial in-plane strain.** No uniaxial strain, no shear, no general
  applied strain tensor.

### Engineering

- **No cluster automation, by design.** No ssh, no submission, no job polling, no
  automatic staging. See [section 7](#7-running-on-an-hpc-cluster) for why.
- **SLURM only.** No PBS/LSF/SGE templates ship, though the interface to adapt is
  three lines ([section 7.6](#76-other-batch-systems)).
- **No resume logic.** Re-running `execute` re-runs every named structure; it does
  not skip ones that already have a good `result-status.json`. Restrict by name
  instead. Each structure is independent and idempotent, so re-running is safe —
  just wasteful.
- **`execute --jobs=N` is thread-based** over independent single-core LAMMPS
  subprocesses. There is no MPI-parallel-per-structure local execution.
- **No CI.** `tests/test_system_agnostic.py` runs anywhere in a bare interpreter,
  but the end-to-end test needs an `lmp` binary, which is why nothing is wired to
  a hosted runner yet.
- **The two-stage fallback is not exercised end-to-end.** Its *decision logic* —
  the convergence test and the coupling-aware pressure gate, shared by both
  runners — is covered by `tests/test_system_agnostic.py` without LAMMPS. But the
  LAMMPS-level machinery (rendering and chaining the alternating fixed-cell and
  box-relax stages) only runs when a structure actually stalls, which the
  well-behaved LJ fixture no longer does. Constructing a fixture that stalls
  *and* recovers, deterministically, is not solved; in practice the path is
  exercised by real runs, where a handful of low-symmetry prototypes take it.
- **No AFLOW API integration.** Prototypes are hand-built in `prototypes.py`
  against the AFLOW encyclopedia's Wyckoff data rather than fetched. `Z1` is not
  implemented (`Z2` and `40` are).
- **Not on PyPI.** Install from source.

## 11. Implementation tips and LAMMPS pitfalls

Hard-won during development. Most of these are general LAMMPS or Python lessons
rather than anything specific to this package.

### Minimisation

- **Never use FIRE or quickmin with `fix box/relax`.** They are incompatible; use
  `min_style cg` (or `sd`). Whenever `box/relax` is active, also set
  `min_modify line quadratic` — the default line search interacts badly with the
  box degrees of freedom and will stall.
- **`ftol` is the raw global 3N force norm and is therefore size-dependent.**
  Convert your per-DOF criterion inside the input script from `count(all)`, as in
  [section 1](#convergence-convention). A "converged" 864-atom cell at the same raw
  `ftol` as a 4-atom cell is not converged to the same standard.
- **`box/relax` stalls are normal, not exceptional.** Build the two-stage fallback
  (alternate fixed-cell and box-only relaxation) in from the start and *record
  which path each structure took* — three of forty Ni–Cu structures needed it.
- **Match `box/relax` coupling to the symmetry.** `aniso` on a hexagonal or
  trigonal cell silently breaks `a = b`; use `tri`. Gate the pressure only on the
  components the coupling actually relaxes, and record the rest as a diagnostic
  instead of failing runs for shear you never asked to relax.

### File I/O

- **Always `write_data … nocoeff`.** For pair styles that support coefficient
  output (`lj/cut` and friends) `write_data` emits a `Pair Coeffs` section; a
  follow-up input that does `read_data` *before* `pair_style` then dies with
  "Must define pair_style before Pair Coeffs". Whether it bites depends on the pair
  style — `eam/alloy` and `pace` write no such section — so the bug stays invisible
  until someone runs a different potential, e.g. in an LJ-based test.
- **Never override `write_data`'s default precision.** The round-trip through the
  file is what the equivalence gate tests; reducing precision breaks it silently.
- **Write the relaxed structure, then re-read it and re-evaluate.** The equivalence
  gate costs a `run 0` and buys you certainty that every number you publish came
  out of LAMMPS.

### Templates and renderers

- **Write placeholder names in lowercase inside `# NEED TO SET:` comments.** A
  string-replace renderer substitutes *all* occurrences of an ALL-CAPS token,
  including the ones in your documentation comment. With a single-line value that
  is merely ugly; with a multi-line value (two `pair_coeff` lines, say) the second
  line lands *outside* the comment and executes. The renderer matches exact
  upper-case tokens, so lowercase names in the comment are inert.
- **Fail loudly on unsubstituted placeholders.** `render()` scans the output for
  known tokens and raises rather than handing LAMMPS a script containing the word
  `MAXITER`.

### Species maps

- **Verify the type map once, deliberately, with the map swapped.** For both
  `eam/alloy` and `pace` it is the element list on the `pair_coeff` line — not the
  file's internal order — that binds types to elements. Run a pure cell with the
  correct map and again with the map reversed; the energies must differ by roughly
  an eV per atom. An inverted map produces perfectly plausible, entirely wrong
  numbers all the way to publication.

### Shell and platform

- **zsh does not word-split unquoted variables.** A loop like
  `for map in "Ni Cu" "Cu Ni"; do set -- $map; …` passes `"Ni Cu"` as *one*
  argument in zsh (unlike bash), and LAMMPS then reads `-var E1 Ni Cu` as a
  multi-value index variable with confusing downstream results. Use explicit
  per-case commands, or `set -- ${=map}` in zsh.
- **`file -i` means different things on Linux and macOS.** On macOS it means "do
  not classify"; the Linux meaning is `file -I` / `--mime`. A charset lint written
  against GNU `file` reports failure on every file on a Mac. `file --mime` works on
  both.
- **Check `src/version.h`, not the directory name,** when you think you know which
  LAMMPS you built.

### Working practice

- **Smoke-test the template before the batch.** A 10-iteration run on the simplest
  structure catches template errors in seconds; a full batch catches them in an
  hour.
- **Probe with one job before releasing an array.** Same rule, cluster-shaped.
- **Never silently truncate.** When a plot clips a range, say so in the panel title
  with the number of clipped entries and the true maximum. When a candidate is
  skipped, write the reason to a file. A missing entry that looks like an absent
  result is worse than an ugly plot.
- **Record choices as choices.** "Cube-on-cube (choice, not optimum)" appears in
  every phase-4 sidecar for a reason: the next reader cannot tell an assumption
  from a result unless you label it.
- **Verify surprises, do not fix them.** Perturbed restarts distinguish a
  relaxation artefact from a property of the potential in about a minute. If it
  reconverges, report it.
- **A truncated pair potential cannot be minimised to a tight tolerance.**
  `lj/cut` (and any hard-cutoff style) makes the lattice energy discontinuous in
  the cell size: each neighbour shell crossing the cutoff steps the energy by tens
  of meV/atom and the pressure by thousands of bar. A `box/relax` line search will
  not climb through that step, so a cell that starts on the wrong side of one
  simply cannot reach its equilibrium, however many fallback cycles you allow.
  Diagnose it by listing the neighbour-shell radii against `rc` before blaming the
  minimiser. Tabulated EAM and ML potentials go smoothly to zero at their cutoff
  and are immune — which is why this only ever bites in test fixtures.
- **Classify by label, never by name.** `name.startswith("vac_")` and
  `name == "fcc_Ni_6x6x6_864at"` both worked perfectly on the system they were
  written for and would have silently mis-handled the next one. Decide what an
  entry *is* from its prototype label and from what the builder recorded in the
  sidecar — that is what `hull0k/labels.py` exists for. The same goes for
  directory names: derive them from the model, do not spell them out.

---

## 12. Validation status

Validated end to end on **Ni–Cu** with two potentials, an EAM parameterisation and
an ACE parameterisation, on 2026-08-26.

| phase | result |
|---|---|
| 0 environment audit | PASS — binary, packages, species map verified in both directions |
| 1 structure pool | PASS — 40 candidates: 6 end members, 21 binary prototypes, 4 MP-only entries (2 further merged as provenance), 3 SQS (108 atoms), 6 defect/reference cells |
| 2 relaxation | PASS — 40/40 per potential; **equivalence gate = 0.0 exactly for all 80 runs**; no symmetry drift anywhere; fallback path taken and recorded for 3 (EAM) and 2 (ACE) structures |
| 3 hull + analysis | PASS with findings — pymatgen and scipy hulls agree for both potentials |
| 4 strained hull | PASS with findings — both potentials, ε = −4 … +4 % with 0.5 % refinement; **354/354 epitaxial relaxations converged** |

**Quantitative gates.** Against on-record 0 K values for the ACE potential: fcc Ni
and fcc Cu lattice constants and cohesive energies agree to
max |Δa₀| = 1.4 × 10⁻⁷ Å and |ΔE| = 3 × 10⁻⁹ eV — far inside the 10⁻³ Å / 1 meV
requirement. The Ni vacancy formation energy comes out at 1.5050 eV in a 256-site
cell against 1.5060 eV on record for a 4000-site cell; dilute Cu-in-Ni
substitution gives 0.2336/0.2337 eV (256/864 sites) against 0.2339 eV on record.

**Findings, not artefacts.** Neither potential reproduces the expected trivial
`{Ni, Cu}` hull, and in both cases perturbed restarts reconverge bit-identically:

- the EAM parameterisation places the `40` (I4₁/amd) NiCu superstructure at
  `E_f = −6.0 meV/atom`, **on** the hull;
- the ACE parameterisation makes dilute Ni-in-Cu substitution **exothermic**
  (`E_sub = −47.8 meV`), so the hull dips below zero in the dilute Cu-rich limit,
  while `E_sub(Cu in Ni) = +0.234 eV` — an asymmetric dilute limit. The x = 0.75
  SQS sits at +10.5 meV/atom, so the negative pocket is confined to the dilute end.

Under the epitaxial constraint on a Ni substrate, the EAM `40` phase enters the
coherent hull between +2.5 % and +3.0 % strain, while the ACE dilute pocket
survives at **every** ε on the grid.

Both are statements about the potentials, and both are exactly the kind of thing
this tool exists to surface.

**Cross-path validation.** The decoupled `prepare → execute → harvest` workflow was
verified bit-identical to the integrated runner on representative cases covering
every code path — a simple cubic cell, a structure that takes the fallback, a
triclinic cell and a 108-atom SQS — and the equivalence is asserted in
`tests/test_fake_end_to_end.py` on every test run.

---

## 13. Citing, licence, author

### Citing

If `hull0k` contributes to published work, please cite it via `CITATION.cff`
(GitHub renders a "Cite this repository" button from it), and cite the potential
you used — the citation string is carried in the config and reproduced in every
`RESULTS_*.md`. Reported energies belong to a specific potential file; the SHA-256
in the hull table header identifies it unambiguously.

Please also cite the tools that do the heavy lifting: **LAMMPS**
(Thompson *et al.*, Comp. Phys. Comm. **271**, 108171, 2022), **ASE**
(Larsen *et al.*, J. Phys.: Condens. Matter **29**, 273002, 2017), **pymatgen**
(Ong *et al.*, Comput. Mater. Sci. **68**, 314, 2013), **spglib**
(Togo & Tanaka, arXiv:1808.01590), and — if you generated SQS cells — **icet**
(Ångqvist *et al.*, Adv. Theory Simul. **2**, 1900015, 2019).

### Contributing

See `CONTRIBUTING.md`. In short: keep system-specific values in configs and out of
code, keep the hull-table schema stable, add new orientations as epitaxy-model
registry entries rather than branches in the driver, and make sure
`tests/test_fake_end_to_end.py` passes.

### Licence and third-party components

`hull0k` is **BSD 3-Clause** (see `LICENSE`). No third-party code is vendored
into this repository; every dependency is installed separately from PyPI.

| dependency | licence | note |
|---|---|---|
| ase | LGPL-2.1-or-later | imported, never modified or bundled |
| icet *(extra)* | MPL-2.0 | file-level copyleft; no icet files are shipped here |
| pymatgen, PyYAML, monty | MIT | |
| spglib, numpy, scipy, mp-api | BSD-3 variants | |
| matplotlib | matplotlib licence (PSF-derived) | BSD-compatible |

Two consequences worth keeping in mind if you fork or extend this:

- **Do not vendor a modified `ase` or any `icet` source file into the tree.**
  Both licences attach obligations to *their own* files; importing carries none,
  copying does.
- **LAMMPS is GPL-2.0, and `hull0k` deliberately keeps it at arm's length.**
  The workflow writes an input script, runs the `lmp` **binary as a subprocess**,
  and parses the log — separate programs communicating at arm's length, so the
  GPL does not reach this code, and no LAMMPS binary is redistributed here.
  **Never replace that with `from lammps import lammps`**: the Python module
  links libLAMMPS in-process, and a distributed combined work would then fall
  under GPL-2.0. The subprocess boundary is a licensing decision as well as a
  scientific one.

### Attribution for data you pull in

- **Materials Project** entries fetched by `hull0k.mp_pool` — structures, MP ids
  and DFT formation energies — are distributed under **CC-BY 4.0**, and the API
  has its own terms of use. The `Ef_DFT_MP_eV` column of every hull table, and
  the faint grey markers in every hull plot, are that data. If you publish either,
  cite the Materials Project (Jain *et al.*, APL Materials **1**, 011002, 2013).
  Provenance is recorded per entry (`mp_id`, `mp_url`) in the pool sidecars, and
  the hull table states the source in its header.
- **Prototype structures** in `hull0k/prototypes.py` are built from Wyckoff data
  in the **AFLOW prototype encyclopedia** (Mehl *et al.*, Comput. Mater. Sci.
  **136**, S1, 2017; Hicks *et al.*, Comput. Mater. Sci. **161**, S1, 2019) —
  including the β-Cu₃Ti cell used for `D0_a`. Crystallographic parameters are
  measurements rather than creative expression, so this is a citation, not a
  licence condition; cite it if the prototype set matters to your result.
- **The potentials are not in this repository** — only their paths and SHA-256
  checksums in the config. Each potential carries its own licence and citation;
  the citation string travels through the config into every `RESULTS_*.md`.

### Author

Erik Bitzek, Max-Planck-Institut für Nachhaltige Materialien (MPI-SusMat),
Düsseldorf. Developed with the LLM-LAMMPS framework.

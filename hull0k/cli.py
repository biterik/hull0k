"""Console entry point: `hull0k <subcommand> <config> [args...]`.

Thin dispatcher over the `python -m hull0k.<module>` entry points, so that
an installed package offers one command instead of nine module paths.
Every subcommand keeps its own argument handling; this only routes.

  hull0k build_pool configs/Ni-Cu.yaml
  hull0k relax      configs/Ni-Cu.yaml EAM-Fischer2018
  hull0k hull       configs/Ni-Cu.yaml EAM-Fischer2018

is identical to

  python -m hull0k.build_pool configs/Ni-Cu.yaml
  python -m hull0k.relax      configs/Ni-Cu.yaml EAM-Fischer2018
  python -m hull0k.hull       configs/Ni-Cu.yaml EAM-Fischer2018

Author: Erik Bitzek (created with LLM-LMPS)
"""

import importlib
import sys

SUBCOMMANDS = {
    "build_pool":  ("phase 1", "build the candidate structure pool"),
    "mp_pool":     ("phase 1c", "add Materials Project entries + DFT column"),
    "relax":       ("phase 2", "integrated relaxation runner (render + run + check)"),
    "prepare":     ("phase 2", "decoupled step 1: render a self-contained RELAX tree"),
    "execute":     ("phase 2", "decoupled step 2: run the prepared tree locally"),
    "harvest":     ("phase 2", "decoupled step 3: integrity checks + relax summary"),
    "verify":      ("any", "re-relax one candidate from perturbed starts"),
    "hull":        ("phase 3", "hull, chemical potentials, defect energies, plots"),
    "report":      ("phase 3", "RESULTS_<potential>.md"),
    "strain":      ("phase 4", "epitaxial relaxation sweep over the strain grid"),
    "strain_hull": ("phase 4", "strained-hull analysis, stability map, crossings"),
}


def usage(stream=sys.stdout):
    from . import __version__
    print(f"hull0k {__version__} -- 0 K convex hulls for binary alloys from "
          f"LAMMPS potentials\n", file=stream)
    print("usage: hull0k <subcommand> <config.yaml> [args...]\n", file=stream)
    width = max(len(s) for s in SUBCOMMANDS)
    for name, (phase, what) in SUBCOMMANDS.items():
        print(f"  {name:<{width}}  [{phase:>8}]  {what}", file=stream)
    print("\nEach subcommand is also runnable as `python -m hull0k.<subcommand>`.",
          file=stream)
    print("See the README for the full manual.", file=stream)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(__version__)
        return 0
    sub = argv[0]
    if sub not in SUBCOMMANDS:
        print(f"hull0k: unknown subcommand '{sub}'\n", file=sys.stderr)
        usage(sys.stderr)
        return 2
    mod = importlib.import_module(f".{sub}", package="hull0k")
    return mod.main(argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())

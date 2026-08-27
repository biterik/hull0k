"""Prototype crystal structures for binary A-B systems.

Every builder returns an ase.Atoms object with chemical symbols assigned;
"A" is the majority/first species, "B" the second. Lattice parameters are
initial guesses only -- phase 2 relaxes everything. Each structure also
carries .info fields: prototype label, box_coupling class for
fix box/relax (per brief section 4), and a source note.

Conventions: cells are conventional cells; decorated-fcc superstructures
are built by plane decoration of fcc supercells and verified downstream
with spglib (the sidecar records the spacegroup *as built*).

Author: Erik Bitzek (created with LLM-LMPS), 2026-08-26
"""

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal


def _mk(cell, frac_species, info):
    symbols = [s for s, _ in frac_species]
    scaled = [p for _, p in frac_species]
    at = Atoms(symbols=symbols, scaled_positions=scaled, cell=cell, pbc=True)
    at.info.update(info)
    return at


# ----------------------------------------------------------------------
# elemental structures
# ----------------------------------------------------------------------

def fcc(el, a):
    fs = [(el, (0, 0, 0)), (el, (0.5, 0.5, 0)), (el, (0.5, 0, 0.5)), (el, (0, 0.5, 0.5))]
    return _mk([a, a, a], fs, dict(prototype="A1_fcc", box_coupling="iso"))


def bcc(el, a):
    fs = [(el, (0, 0, 0)), (el, (0.5, 0.5, 0.5))]
    return _mk([a, a, a], fs, dict(prototype="A2_bcc", box_coupling="iso"))


def hcp(el, a, covera=np.sqrt(8.0 / 3.0)):
    c = covera * a
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    fs = [(el, (1 / 3, 2 / 3, 1 / 4)), (el, (2 / 3, 1 / 3, 3 / 4))]
    return _mk(cell, fs, dict(prototype="A3_hcp", box_coupling="tri"))


# ----------------------------------------------------------------------
# binary prototypes, explicit conventional cells
# A = majority species (3 of 4 in A3B), B = minority
# ----------------------------------------------------------------------

def L12(A, B, a):
    fs = [(B, (0, 0, 0)), (A, (0.5, 0.5, 0)), (A, (0.5, 0, 0.5)), (A, (0, 0.5, 0.5))]
    return _mk([a, a, a], fs, dict(prototype="L1_2", box_coupling="iso"))


def L10(A, B, a, c=None):
    c = a if c is None else c
    fs = [(A, (0, 0, 0)), (A, (0.5, 0.5, 0)), (B, (0.5, 0, 0.5)), (B, (0, 0.5, 0.5))]
    return _mk([a, a, c], fs, dict(prototype="L1_0", box_coupling="aniso"))


def B2(A, B, a):
    fs = [(A, (0, 0, 0)), (B, (0.5, 0.5, 0.5))]
    return _mk([a, a, a], fs, dict(prototype="B2", box_coupling="iso"))


def B1(A, B, a):
    fcc_s = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    oct_s = [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)]
    fs = [(A, p) for p in fcc_s] + [(B, p) for p in oct_s]
    return _mk([a, a, a], fs, dict(prototype="B1", box_coupling="iso"))


def B3(A, B, a):
    fcc_s = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    tet_s = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)]
    fs = [(A, p) for p in fcc_s] + [(B, p) for p in tet_s]
    return _mk([a, a, a], fs, dict(prototype="B3", box_coupling="iso"))


def D022(A, B, a, c=None):
    """Al3Ti type, I4/mmm; conventional bct cell, c ~ 2a. A3B."""
    c = 2.0 * a if c is None else c
    fs = [
        (B, (0, 0, 0)), (B, (0.5, 0.5, 0.5)),
        (A, (0, 0, 0.5)), (A, (0.5, 0.5, 0)),
        (A, (0, 0.5, 0.25)), (A, (0.5, 0, 0.25)),
        (A, (0, 0.5, 0.75)), (A, (0.5, 0, 0.75)),
    ]
    return _mk([a, a, c], fs, dict(prototype="D0_22", box_coupling="aniso"))


def C11b(A, B, a, c=None, z=1.0 / 3.0):
    """MoSi2 type, I4/mmm; A = the '2' species (Si), B = the '1' (Mo). AB2 -> A2B."""
    c = 3.0 * a if c is None else c
    fs = [
        (B, (0, 0, 0)), (B, (0.5, 0.5, 0.5)),
        (A, (0, 0, z)), (A, (0, 0, 1 - z)),
        (A, (0.5, 0.5, 0.5 + z)), (A, (0.5, 0.5, 0.5 - z)),
    ]
    return _mk([a, a, c], fs, dict(prototype="C11_b", box_coupling="aniso"))


def A15(A, B, a):
    """Cr3Si type, Pm-3n. A3B: A on the 6c chains, B on 2a."""
    fs = [
        (B, (0, 0, 0)), (B, (0.5, 0.5, 0.5)),
        (A, (0.25, 0, 0.5)), (A, (0.75, 0, 0.5)),
        (A, (0.5, 0.25, 0)), (A, (0.5, 0.75, 0)),
        (A, (0, 0.5, 0.25)), (A, (0, 0.5, 0.75)),
    ]
    return _mk([a, a, a], fs, dict(prototype="A15", box_coupling="iso"))


def C15(A, B, a):
    """Cu2Mg Laves, Fd-3m (origin choice 2: B on 8a at 1/8,1/8,1/8; A on 16d
    at 1/2,1/2,1/2). A2B. Built via ase.spacegroup.crystal Wyckoff expansion."""
    at = crystal(
        symbols=[B, A],
        basis=[(0.125, 0.125, 0.125), (0.5, 0.5, 0.5)],
        spacegroup=227,
        setting=2,
        cellpar=[a, a, a, 90, 90, 90],
    )
    at.info.update(dict(prototype="C15", box_coupling="iso"))
    return at


def D019(A, B, a, covera=np.sqrt(8.0 / 3.0)):
    """Ni3Sn / Mg3Cd type, P6_3/mmc. A3B. a ~ 2*a_hcp. Wyckoff: B 2c, A 6h x=5/6."""
    c = covera * a / 2.0  # hcp c of the underlying lattice; cell c equals it
    at = crystal(
        symbols=[B, A],
        basis=[(1 / 3, 2 / 3, 0.25), (5 / 6, 2 / 3, 0.25)],
        spacegroup=194,
        cellpar=[a, a, c, 90, 90, 120],
    )
    at.info.update(dict(prototype="D0_19", box_coupling="tri"))
    return at


def D0a(A, B, a=None, b=None, c=None, scale=1.0):
    """beta-Cu3Ti type, Pmmn (#59, origin choice 2). A3B.
    Wyckoff (AFLOW prototype encyclopedia, beta-Cu3Ti):
      B  2b (1/4, 3/4, z1), z1 ~ 0.660
      A  2a (1/4, 1/4, z2), z2 ~ 0.155
      A  4f (x, 1/4, z3),  x ~ 0.508? -- AFLOW uses (x,1/4,z) 4f
    Cell scaled from beta-Cu3Ti a=5.162, b=4.347, c=4.531 by `scale`.
    """
    a0, b0, c0 = 5.162, 4.347, 4.531
    a = a0 * scale if a is None else a
    b = b0 * scale if b is None else b
    c = c0 * scale if c is None else c
    at = crystal(
        symbols=[B, A, A],
        basis=[(0.25, 0.75, 0.660), (0.25, 0.25, 0.155), (0.008, 0.25, 0.658)],
        spacegroup=59,
        setting=2,
        cellpar=[a, b, c, 90, 90, 90],
    )
    at.info.update(dict(prototype="D0_a", box_coupling="aniso"))
    return at


# ----------------------------------------------------------------------
# decorated-fcc superstructures (plane decoration of an fcc supercell)
# ----------------------------------------------------------------------

def _fcc_supercell_sites(nx, ny, nz):
    """fcc sites of an nx x ny x nz conventional supercell, in units of a/2
    (integer triples with even component sum)."""
    sites = []
    for i in range(2 * nx):
        for j in range(2 * ny):
            for k in range(2 * nz):
                if (i + j + k) % 2 == 0:
                    sites.append((i, j, k))
    return sites


def L11(A, B, a):
    """CuPt type, R-3m: fcc with (111) layers alternating A,B.
    Built in the hexagonal setting: 6 (111) layers along c."""
    ah = a / np.sqrt(2.0)
    ch = 6.0 * a / np.sqrt(3.0)
    cell = [[ah, 0, 0], [-ah / 2, ah * np.sqrt(3) / 2, 0], [0, 0, ch]]
    offsets = [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]  # A,B,C stacking
    fs = []
    for m in range(6):
        ox, oy = offsets[m % 3]
        sp = A if m % 2 == 0 else B
        fs.append((sp, (ox, oy, m / 6.0)))
    return _mk(cell, fs, dict(prototype="L1_1", box_coupling="tri"))


def Z2(A, B, a):
    """A2B2 [001] superlattice ('Z2'): 1x1x2 fcc supercell, species by
    (001) plane index k in units of a/2: k mod 4 in {0,1} -> A else B."""
    fs = []
    for (i, j, k) in _fcc_supercell_sites(1, 1, 2):
        sp = A if (k % 4) < 2 else B
        fs.append((sp, (i / 2.0, j / 2.0, k / 4.0)))
    return _mk([a, a, 2 * a], fs, dict(prototype="Z2", box_coupling="aniso"))


def S40(A, B, a):
    """'40' (NbP-like) A2B2 superlattice along [201]: 1x1x2 fcc supercell,
    species by plane index n = 2i + k (units a/2): n mod 4 in {0,1} -> A."""
    fs = []
    for (i, j, k) in _fcc_supercell_sites(1, 1, 2):
        n = 2 * i + k
        sp = A if (n % 4) < 2 else B
        fs.append((sp, (i / 2.0, j / 2.0, k / 4.0)))
    return _mk([a, a, 2 * a], fs, dict(prototype="40", box_coupling="tri"))

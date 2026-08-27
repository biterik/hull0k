"""Prototype-label vocabulary: how the analysis layer classifies pool entries.

The pool builder writes a `prototype` label into every sidecar and into the
hull table. Downstream code must decide from that LABEL what kind of entry it
is looking at -- never from the entry's NAME. A name test like
`name.startswith("vac_")` or `name == "fcc_Ni_6x6x6_864at"` silently stops
working the moment the elements, the host lattice or the supercell size
change; a label test does not.

The vocabulary the builders emit (hull0k/build_pool.py, hull0k/prototypes.py):

    A1_fcc, A2_bcc, A3_hcp, L1_2, L1_0, B1, B2, B3, D0_22, D0_19, D0_a,
    C11_b, C15, A15, L1_1, Z2, 40      ordinary bulk candidates
    <parent>_supercell                  perfect-host finite-size reference
    vacancy_in_<lattice>                vacancy cell
    substitutional_in_<lattice>         dilute substitution cell
    SQS_<parent>                        special quasirandom structure
    MP:<formula>                        Materials Project entry

Keep these predicates as the single place that knows those spellings.

Author: Erik Bitzek (created with LLM-LMPS)
"""

VACANCY_PREFIX = "vacancy_in_"
SUBSTITUTIONAL_PREFIX = "substitutional_in_"
SQS_PREFIX = "SQS_"
SUPERCELL_SUFFIX = "_supercell"


def is_vacancy(prototype):
    return str(prototype).startswith(VACANCY_PREFIX)


def is_substitutional(prototype):
    return str(prototype).startswith(SUBSTITUTIONAL_PREFIX)


def is_defect(prototype):
    """A point-defect cell: vacancy or dilute substitution."""
    return is_vacancy(prototype) or is_substitutional(prototype)


def is_sqs(prototype):
    """A special quasirandom structure on any parent lattice."""
    return str(prototype).startswith(SQS_PREFIX)


def is_supercell_reference(prototype):
    """A perfect-host supercell kept only to quantify a finite-size shift.

    Energetically degenerate with its primitive end member, so it must not
    appear in the metastability spectrum as if it were a separate candidate.
    """
    return str(prototype).endswith(SUPERCELL_SUFFIX)


def is_bulk_candidate(prototype):
    """True for entries that belong in the metastability spectrum: ordinary
    bulk structures and solid solutions, but not defect cells and not
    finite-size references."""
    return not (is_defect(prototype) or is_supercell_reference(prototype))

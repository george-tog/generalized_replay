"""Figure labels for animals and sessions, one fixed mapping per dataset.

The kernel and parameter labels are in replay_hmm.plots.labels.
"""

# The pooled row/partition label of the best-kernel share tables and the two-map jump pools.
POOLED_LABEL = "ALL"


# Anonymised animal labels for figures, shared by every figure so an animal is named the same
# way everywhere. Deliberately a fixed dict rather than "Rat N in sorted order": deriving it
# would renumber the animals whenever a run covers a subset (e.g. --rats Ned TinyTim), so the
# same figure label would mean different rats across runs.
#   Lucky -> Rat 1, Ned -> Rat 2, TinyTim -> Rat 3
#
# The other four take 4-7, ordered LCRC first and LCR last, alphabetically within task, so
# the two task groups are contiguous on any axis sorted by label: BB and Spot ran LCRC like
# the first three, while Huey and Stinky ran the LCR rotation.
#   BB -> Rat 4, Spot -> Rat 5 (LCRC);  Huey -> Rat 6, Stinky -> Rat 7 (LCR)
# Note BUZSAKI_RAT_DISPLAY_NAMES below also uses "Rat 4"; the two datasets never share an
# axis, and deriving either from the other would renumber animals when a run covers a subset.
RAT_DISPLAY_NAMES = {
    "Lucky": "Rat 1",
    "Ned": "Rat 2",
    "TinyTim": "Rat 3",
    "BB": "Rat 4",
    "Spot": "Rat 5",
    "Huey": "Rat 6",
    "Stinky": "Rat 7",
}


def rat_label(rat: str) -> str:
    """Figure label for an animal: the anonymised name, or the real one if unmapped.

    Falling back to the real name keeps an unrecognised animal visible rather than
    silently collapsing it into someone else's label.
    """
    return RAT_DISPLAY_NAMES.get(str(rat), str(rat))


# Buzsaki animals, kept separate from RAT_DISPLAY_NAMES rather than merged into it: the
# two datasets are never plotted on one axis, and a single dict would have to choose
# whether "Rat 1" means Lucky or Achilles. Fixed rather than derived for the same reason
# as above -- a run covering a subset must not renumber the animals.
#   Achilles -> Rat 1, Buddy -> Rat 2, Cicero -> Rat 3, Gatsby -> Rat 4
BUZSAKI_RAT_DISPLAY_NAMES = {
    "Achilles": "Rat 1",
    "Buddy": "Rat 2",
    "Cicero": "Rat 3",
    "Gatsby": "Rat 4",
}


def buzsaki_rat_label(rat: str) -> str:
    """Figure label for a Buzsaki animal; the real name if unmapped, as with rat_label."""
    return BUZSAKI_RAT_DISPLAY_NAMES.get(str(rat), str(rat))


def buzsaki_session_label(session: str) -> str:
    """Figure label for a Buzsaki session: animal anonymised, date kept.

    'Achilles_10252013' -> 'Rat 1_10252013'; a bare 'Achilles' -> 'Rat 1'. The date is
    kept because two Cicero sessions would otherwise collapse to the same panel title.

    Anything whose leading token is not a known animal passes through untouched, so the
    pooled labels ('ALL', 'ALL_excl_low_yield') and the direction keys ('out', 'in') are
    safe to route through here, and an unrecognised animal stays visible rather than
    silently taking someone else's label.
    """
    animal, separator, rest = str(session).partition("_")
    if animal not in BUZSAKI_RAT_DISPLAY_NAMES:
        return str(session)
    return BUZSAKI_RAT_DISPLAY_NAMES[animal] + separator + rest

# Pfeiffer's animals are already anonymous in the source data (directories Rat1..Rat5). This
# mapping numbers the linear-track cohort contiguously (Rat3 and Rat4 ran only the open
# field), and the space marks a mapped label: an unmapped animal falls through as "Rat3",
# visibly different from a mapped "Rat 3". Fixed rather than derived, as above.
#   Rat1 -> Rat 1, Rat2 -> Rat 2, Rat5 -> Rat 3
PFEIFFER_RAT_DISPLAY_NAMES = {
    "Rat1": "Rat 1",
    "Rat2": "Rat 2",
    "Rat5": "Rat 3",
}


def pfeiffer_rat_label(rat: str) -> str:
    """Figure label for a Pfeiffer animal; the directory name if unmapped."""
    return PFEIFFER_RAT_DISPLAY_NAMES.get(str(rat), str(rat))


def pfeiffer_session_label(rat: str, session: str) -> str:
    """Figure label for a Pfeiffer session: 'Rat 1 Linear3'.

    Unlike Buzsaki, the session name alone is not unique -- 'Linear1' occurs under three
    animals -- so the animal is always part of the label.
    """
    return f"{pfeiffer_rat_label(rat)} {session}"

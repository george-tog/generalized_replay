"""Display labels and colours for kernels, kernel families and transition parameters.

The animal and session labels live in replay_hmm.datasets.display.
"""

KERNEL_LABELS = {
    "sticky_TCMSR": "TCM-SR",
    "sticky_TCMSR_sym": "Sym. TCM-SR",
    "sticky_SR": "SR",
    "sticky_SR_rev": "Rev. SR",
    "movement_empirical": "Movement",
    "continuous_EM": "Continuous",
    "momentum": "Momentum",
    "stationary": "Stationary",
    "fragmented": "Fragmented",
    # Uniform within a map with kappa and eps fitted: the sigma -> infinity limit of
    # continuous_EM. Not in KERNEL_ORDER; listed so the CV and BIC strips print a label rather
    # than the raw snake_case name, which is what kernel_label's fallback would give.
    "segment_uniform": "Seg. uniform",
}

def kernel_label(kernel: str) -> str:
    return KERNEL_LABELS.get(kernel, kernel)


# Labels for the LUMPED four-way figure (hmm.kernel_sets.SUBSET_KERNEL_GROUPS), whose segments
# are keyed on a representative kernel but stand for that kernel's whole family. Kept separate
# from KERNEL_LABELS rather than merged into it: the same key means one kernel in every other
# figure and a family only here, so a single dict would have to pick one and mislabel the
# other. Naming the absorbed sibling in the label is the point -- a bare "TCM-SR" on a bar
# that also counts Sym. TCM-SR's wins is exactly the misreading this figure invites.
SUBSET_GROUP_LABELS = {
    "sticky_TCMSR": "TCM-SR\n+ Sym.",
    "continuous_EM": "Continuous\n+ Momentum",
}


def subset_group_label(kernel: str) -> str:
    """Family label for the lumped four-way figure; the plain kernel label if it stands alone.

    `fragmented` and `stationary` are one-member groups, so they fall through to
    `kernel_label` and read identically in both four-way figures.
    """
    return SUBSET_GROUP_LABELS.get(kernel, kernel_label(kernel))


# Okabe & Ito (2008) colourblind-safe qualitative palette. Fixed kernel -> hue so a kernel
# keeps its colour across every figure and every dataset. Not COLOR_PALETTE: the house palette
# cannot supply this many separable hues (blue<->purple deltaE 4.8 deutan).
#
# THE single source: every figure takes kernel hues from here. Every kernel in
# hmm.kernel_sets.KERNEL_ORDER needs an entry (tests/test_lustig_event_kernel_cache.py checks).
#
# Ten kernels, and Okabe & Ito supplies only six usable slots -- #F0E442 yellow falls below
# the lightness floor against white and #000000 black would read as an axis -- so four entries
# are not unmodified palette hex. Two of them darken a sibling's hue rather than inventing a
# new one, which is a second, non-colour channel (lightness) that survives CVD:
#
#   sticky_TCMSR_sym  #004C6D, TCM-SR's blue darkened. It is the nested omega=1 variant of
#                     TCM-SR, so reading as "same family, darker" is the correct message.
#   segment_uniform   #006E50, continuous_EM's bluish green darkened, by the same rule: it is
#                     continuous_EM's nested sigma -> infinity limit.
#   momentum          #882255, wine. A genuinely new hue, because it is NOT nested inside
#                     continuous_EM in the darker-sibling sense -- it adds an order rather
#                     than removing a parameter -- and because the reddish purple that would
#                     otherwise pair it with continuous_EM already belongs to reverse SR.
#   fragmented        #999999, neutral grey, leaving the reddish purple to reverse SR so no
#                     kernel changes colour between the direction ballot and the full one. A
#                     parameter-free null reading as neutral rather than as a saturated hue is
#                     a better encoding anyway.
#
# Worst CVD pair (reddish purple <-> bluish green) and worst normal-vision pair (orange <->
# vermillion) are both non-adjacent in the stack order. Hue separation sits in the 6-8 deltaE
# band under simulated deutan/protan vision, which is only legible with a second, non-colour
# cue: the printed percentage in every segment, backed by the white gap between segments. Those
# numbers are load-bearing, not decoration, because the two sibling pairs (TCM-SR / symmetric,
# SR / reverse) are deliberately adjacent.
KERNEL_COLORS = {
    "sticky_TCMSR": "#0072B2",        # blue
    "sticky_TCMSR_sym": "#004C6D",    # darkened blue -- TCM-SR's nested omega=1 variant
    "sticky_SR": "#56B4E9",           # sky blue -- same family as TCM-SR, its sibling kernel
    "sticky_SR_rev": "#CC79A7",       # reddish purple -- its hue in the direction figures
    "continuous_EM": "#009E73",       # bluish green
    "momentum": "#882255",            # wine -- distinct from the reddish purple of Rev. SR
    "movement_empirical": "#E69F00",  # orange
    "fragmented": "#999999",          # neutral grey -- a parameter-free null
    "stationary": "#D55E00",          # vermillion
    # See the block above. Not drawn by the strip or share figures -- the CV, BIC and
    # fold-parameter strips are monochrome and plots.kernel_share runs over KERNEL_ORDER, which
    # this kernel is off -- but registered so the hue is claimed and stays distinct from every
    # other kernel's.
    "segment_uniform": "#006E50",    # darkened bluish green -- continuous_EM's nested limit
}


# Mathtext symbols for the HMM transition parameters so figure axes read as Greek
# letters rather than the raw column stems. Keyed on the bare stem; the ``grid_``
# prefixed generating-parameter columns are stripped before lookup. Lives here
# alongside the kernel labels so every figure spells a parameter the
# same way instead of one plotter drifting from another.
PARAM_MATH_LABELS = {
    "gamma": r"$\gamma$",
    "kappa": r"$\kappa$",
    "eps": r"$\epsilon$",
    "omega": r"$\omega$",
    "sigma_arm": r"$\sigma_{\mathrm{arm}}$",
    "sigma_wheel": r"$\sigma_{\mathrm{wheel}}$",
    # `momentum`'s two axes. Column names carry the prefix -- a bare "a" in a table this wide
    # is asking to collide -- but the axis label is the symbol the model is written in, so the
    # prefix is dropped here rather than in each plotter. The unit is on sigma_v because it is
    # a speed in cm/s and so comparable to a replay speed; `a` is the dimensionless per-bin
    # carry-over exp(-lambda_m dt).
    "momentum_a": r"$a$",
    "momentum_sigma_v_cm_s": r"$\sigma_v$ (cm/s)",
    # The same two axes again under the names the SWEEP columns carry. `momentum_a` above is
    # the fitted value the CV strips read; the recovery rows spell the generating value
    # `grid_a`/`grid_sigma_v`, which strip to these. Without them the pairwise recoverability
    # panel falls through to the raw stem and its axes read "a" and "sigma_v" in plain text
    # beside sibling panels showing Greek.
    "a": r"$a$",
    "sigma_v": r"$\sigma_v$ (cm/s)",
}


def param_math_label(param: str) -> str:
    """Mathtext symbol for a parameter, accepting either the bare or ``grid_`` name."""
    stem = str(param)
    if stem.startswith("grid_"):
        stem = stem[len("grid_"):]
    return PARAM_MATH_LABELS.get(stem, stem)

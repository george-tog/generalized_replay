"""The HMM transition-kernel vocabulary: the comparison ballot and its partitions.

Every module naming a kernel takes the name from here, plotters included. `plots.labels`
does the same for the kernels' labels.
"""
from __future__ import annotations


# Fixed display/stack order, so figure segments never reorder with the data, and fixed
# comparison order, so an argmax over the nine is reproducible. Siblings are adjacent: each
# structured kernel is followed by its nested variant, so a stacked figure reads as four
# families rather than nine unrelated arms.
#
#   sticky_TCMSR      A_local = (1-omega) M + omega (M.T @ M), omega learned
#   sticky_TCMSR_sym  the same with omega pinned at 1, i.e. A_local = M.T @ M -- properly
#                     nested inside sticky_TCMSR at one parameter fewer, so the BIC gap
#                     between them is a direct test of whether omega earns its parameter
#   sticky_SR         A_local = SR(P)
#   sticky_SR_rev     A_local = SR(P.T) == SR(P).T -- the same kernel swept backwards
#   momentum          the ONLY second-order kernel: s_t ~ N((1+a)s_{t-1} - a s_{t-2}, q), a
#                     damped-momentum (Krause et al. 2022) process. Placed next to
#                     continuous_EM because that is what it reduces to at a = 0, and because
#                     the two agree on the jump marginal and differ only in its
#                     autocorrelation -- so they are the pair a reader should compare.
#                     Unlike every other kernel it is not an ssm.HMM: it carries its own
#                     order-2 forward-backward (hmm.momentum) and is fitted by a grid
#                     search over (a, sigma_v), stored as (lambda_m, sigma_m), rather than
#                     by EM.
KERNEL_ORDER = (
    "sticky_TCMSR",
    "sticky_TCMSR_sym",
    "sticky_SR",
    "sticky_SR_rev",
    "continuous_EM",
    "momentum",
    "movement_empirical",
    "fragmented",
    "stationary",
)
# The six-kernel ballot without the symmetrised, reversed and momentum arms: TCM-SR, SR,
# movement, continuous, fragmented, stationary. Kept as a named set rather than an ad-hoc
# tuple in each plotter, so the six-kernel figures across the three datasets cannot drift from
# one another -- the same argument as KERNEL_ORDER above. Derived by exclusion rather than
# spelled out, so it cannot come to name a kernel that no longer exists.
#
# This is a DISPLAY subset for the model-comparison strips, not a partition of the ballot: the
# argmax figures still run over all nine. Do not confuse it with SUBSET_KERNELS (the four-way
# one-representative-per-family set) or with COHERENT_KERNELS, which has seven members -- it
# keeps those three arms and drops the two nulls.
ORIGINAL_SIX_KERNELS = tuple(
    k for k in KERNEL_ORDER if k not in ("sticky_TCMSR_sym", "sticky_SR_rev", "momentum")
)
assert len(ORIGINAL_SIX_KERNELS) == 6, ORIGINAL_SIX_KERNELS

# One representative per transition family -- structured retrieval (TCM-SR), smooth movement
# through space (continuous), and the two parameter-free nulls. Dropping SR and Movement takes
# the two rivals nearest TCM-SR and continuous respectively out of the argmax, so the four-way
# figure answers "which family wins" without the within-family split diluting either share.
#
# NOT derived by exclusion, unlike ORIGINAL_SIX_KERNELS above: it is a deliberate choice of one
# representative per family, so a new kernel joining KERNEL_ORDER must not silently join this.
# Lives here rather than in a plotter so every plotter shares one definition.
SUBSET_KERNELS = ("sticky_TCMSR", "continuous_EM", "fragmented", "stationary")
assert set(SUBSET_KERNELS) <= set(KERNEL_ORDER), SUBSET_KERNELS

# The four-way figure's other reading. SUBSET_KERNELS restricts the argmax to ONE
# representative per family, so `sticky_TCMSR_sym`'s and `momentum`'s wins are not counted at
# all -- which understates those two families exactly when the nested variant is the better
# fit, and both of them routinely are. These groups instead put each family's FULL membership
# on the ballot and lump the winners: the four shares still sum to 100, but a family is
# credited for a win by any of its members.
#
# The two flanking rivals stay off the ballot, as in SUBSET_KERNELS and for the same reason --
# `sticky_SR` and `movement_empirical` are the nearest neighbours of TCM-SR and continuous
# respectively, and leaving them in dilutes both families. So the union here is six kernels,
# not nine.
#
# Keyed on the REPRESENTATIVE kernel, not on a new family name, so the merged figure inherits
# that kernel's hue and stack position: the two four-way figures are meant to be read side by
# side and a family changing colour between them would defeat that. Only the label differs --
# see plots.labels.SUBSET_GROUP_LABELS and plot_best_kernel_stacked's `labeller`.
SUBSET_KERNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "sticky_TCMSR": ("sticky_TCMSR", "sticky_TCMSR_sym"),
    "continuous_EM": ("continuous_EM", "momentum"),
    "fragmented": ("fragmented",),
    "stationary": ("stationary",),
}
# Same four columns in the same order as the restricted figure, or they are not comparable.
assert tuple(SUBSET_KERNEL_GROUPS) == SUBSET_KERNELS, tuple(SUBSET_KERNEL_GROUPS)
# A representative must be a member of its own group, or the inherited colour would name a
# kernel that cannot win that segment.
assert all(k in members for k, members in SUBSET_KERNEL_GROUPS.items())
# Every member must be on the ballot, and no kernel may sit in two groups -- either would make
# the lumped shares stop summing to 100. Checked against KERNEL_ORDER rather than
# COMPARISON_KERNELS, which is defined further down: same tuple, but this block runs first.
SUBSET_GROUP_MEMBERS = tuple(
    k for members in SUBSET_KERNEL_GROUPS.values() for k in members
)
assert set(SUBSET_GROUP_MEMBERS) <= set(KERNEL_ORDER), SUBSET_GROUP_MEMBERS
assert len(set(SUBSET_GROUP_MEMBERS)) == len(SUBSET_GROUP_MEMBERS), SUBSET_GROUP_MEMBERS

# The four-way ballot again, with `momentum` standing for the movement family in place of
# `continuous_EM`. Same question as SUBSET_KERNELS -- which family explains each event -- put
# to the movement family's OTHER representative, and the two figures are meant to be read as a
# pair: the continuous kernel is the a = 0 special case of the momentum one, so the difference
# between the two figures is what the second-order term buys the movement family in the
# argmax, event by event, rather than in a pooled CV or BIC score.
#
# Not a lumping (SUBSET_KERNEL_GROUPS already answers "either movement kernel") and not a
# superset: momentum is the only movement arm on this ballot, so an event it loses to TCM-SR
# is genuinely better explained by structured retrieval than by damped momentum. Adding
# continuous_EM alongside it would split the family and answer neither question.
#
# Position 1, so the movement slot is the same slot in both four-way figures and the other
# three segments keep their stack positions and hues. Spelled out rather than derived from
# SUBSET_KERNELS by substitution, for the reason SUBSET_KERNELS itself is spelled out: a
# representative set is a choice, not a consequence of KERNEL_ORDER.
MOMENTUM_SUBSET_KERNELS = ("sticky_TCMSR", "momentum", "fragmented", "stationary")
assert set(MOMENTUM_SUBSET_KERNELS) <= set(KERNEL_ORDER), MOMENTUM_SUBSET_KERNELS
assert len(MOMENTUM_SUBSET_KERNELS) == len(SUBSET_KERNELS), MOMENTUM_SUBSET_KERNELS
# It must differ from the continuous four-way in the movement slot and ONLY there, or the two
# figures stop being a controlled comparison of the two movement representatives.
assert [
    (a, b) for a, b in zip(SUBSET_KERNELS, MOMENTUM_SUBSET_KERNELS) if a != b
] == [("continuous_EM", "momentum")], MOMENTUM_SUBSET_KERNELS

# No learnable transition parameters, so their transferred LL must equal their self LL
# exactly. That identity is the pipeline's free correctness check.
#
# `momentum` is NOT here, and the resemblance is a trap worth naming: it runs no EM either,
# and `has_learnable_transition_params` is False for it. But its (lambda_m, sigma_m) ARE
# fitted -- by a grid search maximizing training likelihood, which is the same procedure EM
# performs with a different optimizer -- so it has parameters to transfer, it is charged
# p = 2 in BIC, and its transferred LL matches its self LL only once the SELECTED grid point
# has been restored from the fit row.
PARAM_FREE_MODELS = ("stationary", "fragmented")

# Self-transfer must reproduce the stored full-data loglik. Above WARN we print the
# discrepancy; above FAIL the rebuild path is wrong and there is no point continuing.
SELF_TOL_WARN_PER_BIN = 1e-6
SELF_TOL_FAIL_PER_BIN = 1e-3

# Per-event coherence partition. `stationary` (kappa=1, self-transitions only) and
# `fragmented` (eps=1, uniform over all K states) place no coherent spatial trajectory in an
# event; the other seven move probability mass through space.
#
# Every kernel shares the same fixed-Poisson observations and the same fixed uniform initial
# distribution (build_hmm_dict), so a per-event log-likelihood difference between two
# kernels is a difference in transition structure and nothing else. That is what makes
# argmax over these nine usable as a coherence test.
#
# `coherent_margin_per_bin` is max(coherent) - max(incoherent), so a wider coherent set can
# only raise the first term: event counts from a cache built under a narrower ballot (e.g.
# ORIGINAL_SIX_KERNELS, with four coherent kernels) are not comparable with this one's.
COMPARISON_KERNELS = tuple(KERNEL_ORDER)
INCOHERENT_KERNELS = tuple(PARAM_FREE_MODELS)
COHERENT_KERNELS = tuple(k for k in COMPARISON_KERNELS if k not in INCOHERENT_KERNELS)
# A kernel that is in neither class would silently sit out of the filter, so partition or die.
assert set(COHERENT_KERNELS) | set(INCOHERENT_KERNELS) == set(COMPARISON_KERNELS)
assert not (set(COHERENT_KERNELS) & set(INCOHERENT_KERNELS))

# Fitted and scored, but deliberately NOT on the argmax ballot above. `segment_uniform` is
# uniform within a map with kappa and eps fitted -- the sigma -> infinity limit of
# continuous_EM -- and a within-map-uniform kernel places no ordered trajectory in an event,
# so it belongs on the incoherent side rather than the coherent one.
#
# That is precisely why promoting it is NOT a one-line change: COHERENT_KERNELS above is
# derived BY EXCLUSION from PARAM_FREE_MODELS, and this kernel is not parameter-free, so a
# promoted segment_uniform would default to COHERENT and inflate the coherent share with both
# partition asserts still passing.
#
# Lives here rather than in hmm.two_map_kernels because that module imports ssm: the scorers
# and plotters that need to know which kernels are off the ballot must be able to ask without
# pulling in the decode stack.
OFF_BALLOT_KERNELS: tuple[str, ...] = ("segment_uniform",)
# Off the ballot means off the ballot; a kernel in both would make the argmax undefined.
assert not set(OFF_BALLOT_KERNELS) & set(KERNEL_ORDER), OFF_BALLOT_KERNELS

# Every kernel whose per-event log-likelihood is cached. Strictly wider than the ballot: the
# caches store an ll_ column per kernel here, while `best_kernel`, `coherent_margin_per_bin`
# and every share figure argmax over COMPARISON_KERNELS alone.
#
# The two lists being different is the whole point, so scorers must name which one they mean.
# The failure this prevents is silent: a scorer that argmaxed over the fit CLI's DEFAULT_KERNELS
# (which includes the off-ballot kernel) writes a `best_kernel` the share plotters cannot draw,
# and best_kernel_share_table's reindex().fillna(0) drops those events -- stacked bars that
# quietly stop summing to 100% rather than an error.
SCORED_KERNELS = COMPARISON_KERNELS + OFF_BALLOT_KERNELS
assert len(set(SCORED_KERNELS)) == len(SCORED_KERNELS), SCORED_KERNELS


# --------------------------------------------------------------------------- #
# The direction comparison
# --------------------------------------------------------------------------- #
# A second, narrower argmax that asks only which DIRECTION explains an event. TCM-SR's local
# kernel is (1-omega)*M + omega*(M.T @ M) -- a forward SR mixed with a symmetric term -- so a
# TCM-SR win over the full ballot is ambiguous between "events are genuinely bidirectional"
# and "events are unidirectional, half forward and half reverse, and the mixture splits the
# difference". Restricting the argmax to the three directional arms separates the two: the
# second story predicts a roughly even SR / reverse-SR split with TCM-SR winning little.
#
# A strict SUBSET of COMPARISON_KERNELS. The ballot is kept as a separate cache because
# restricting the argmax is the whole point -- over all nine, a
# directionally reversed event can still be won by `continuous_EM` or `movement_empirical`,
# and the direction question would go unanswered.
DIRECTION_KERNELS = ("sticky_TCMSR", "sticky_SR", "sticky_SR_rev")

# Maps a kernel with no fit of its own to the fitted kernel whose transition parameters it
# borrows. Empty: every kernel on every ballot is fitted independently, so the direction
# argmax compares each kernel at its own optimum.
#
# score_events_per_kernel takes `param_donors`; an empty mapping flows through it unchanged.
# A kernel listed here would
# read its parameters from the donor's row and skip the stored-loglik identity check, since
# that stored value belongs to the donor.
PARAM_DONORS: dict[str, str] = {}
# A donor that is not itself scorable would be unreadable from any fit summary.
assert set(PARAM_DONORS.values()) <= set(COMPARISON_KERNELS)
# The direction ballot must be scorable from the same fits as the main one.
assert set(DIRECTION_KERNELS) <= set(COMPARISON_KERNELS) | set(PARAM_DONORS)


def ll_col(kernel: str, kind: str) -> str:
    """Column name for a per-event log-likelihood; kind is "total" or "per_bin"."""
    return f"ll_{kernel}__{kind}"


def ll_columns(kernels: tuple[str, ...] = COMPARISON_KERNELS) -> tuple[str, ...]:
    """Every per-event log-likelihood column, in kernel-then-kind order."""
    return tuple(ll_col(k, kind) for k in kernels for kind in ("total", "per_bin"))

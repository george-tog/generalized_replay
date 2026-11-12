"""Which sessions each analysis runs on, and the event sources each dataset can detect.

Every set is kept under its own name, as its own literal, and never derived from another:
each is what its own analysis runs on, so editing one cannot silently change another's sweep.
The one exception is BUZSAKI_LINEAR_SESSIONS, which is by definition BUZSAKI_SESSIONS without
the circular mazes.
"""

from __future__ import annotations


# ---------------------------------------------------------------- Lustig (Lee lab)
# The CV fits (`replay-hmm submit fit --dataset lustig`).
LUSTIG_SESSIONS: dict[str, list[str]] = {
    "TinyTim": [f"rec{i:02d}" for i in range(1, 7)],
    "Lucky": [f"rec{i:02d}" for i in range(1, 6)],
    "Ned": [f"rec{i:02d}" for i in range(1, 5)],
    # One session each, and all four live in the per-session directory
    # layout (<rat>/rec01/rec01_BehavElectrDataLFP.mat), which `session_mat_path` resolves.
    # Spot, Huey and Stinky are on the other camera calibration, so at `--bin-width-cm 3.5`
    # they resolve to 10 px where everything above resolves to 8.
    "BB": ["rec01"],
    "Spot": ["rec01"],
    "Huey": ["rec01"],
    "Stinky": ["rec01"],
}

# The synthetic recovery runs (`replay-hmm submit recovery`).
LUSTIG_RECOVERY_SESSIONS: dict[str, list[str]] = {
    "TinyTim": [f"rec{i:02d}" for i in range(1, 7)],
    "Lucky": [f"rec{i:02d}" for i in range(1, 6)],
    "Ned": [f"rec{i:02d}" for i in range(1, 5)],
}

# The misspec sweep's cluster array (`replay-hmm submit misspec-sweep`, one task per entry, in
# this order: `misspec sweep --slurm-array session` reads entry $SLURM_ARRAY_TASK_ID). The 14
# sessions with >= 50 fold-0 test bins at the 10 ms source fit (Ned/rec01-03, Spot/rec01 and
# Stinky/rec01 fall below it), as the sweep's "rat/session" --sessions values.
LUSTIG_MISSPEC_SWEEP_SESSIONS: tuple[str, ...] = (
    "BB/rec01", "Huey/rec01", "Lucky/rec01", "Lucky/rec02", "Lucky/rec03", "Lucky/rec04",
    "Lucky/rec05", "Ned/rec04", "TinyTim/rec01", "TinyTim/rec02", "TinyTim/rec03",
    "TinyTim/rec04", "TinyTim/rec05", "TinyTim/rec06",
)

# ---------------------------------------------------------------- Buzsaki (Grosmark & Buzsaki 2016)
# Session registry: (display_name, path_relative_to_BUZSAKI_DIR)
BUZSAKI_SESSIONS: list[tuple[str, str]] = [
    ("Achilles_10252013", "Achilles_10252013/Achilles_10252013/Achilles_10252013_sessInfo.mat"),
    ("Buddy_06272013",    "Buddy_06272013/Buddy_06272013_sessInfo.mat"),
    ("Cicero_09172014",   "Cicero_09172014/Cicero_09172014_sessInfo.mat"),
    ("Cicero_09012014",   "Cicero_09012014/Cicero_09012014_sessInfo.mat"),
    ("Gatsby_08022013",   "Gatsby_08022013/Gatsby_08022013_sessInfo.mat"),
]

# The dataset's circular-maze sessions, left out of every analysis (BUZSAKI_SESSIONS does not list
# them); the filters below and in the Buzsaki commands also drop them by name.
BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS = frozenset({
    "Achilles_11012013",
    "Cicero_09102014",
    "Gatsby_08282013",
})

# The linear-track sessions the Buzsaki HMM fit, its jumps, scorer, PPC and viewer run on:
# BUZSAKI_SESSIONS without the circular mazes (rare reverse-direction travel makes their
# inbound maps sparse). Derived, not its own literal: it is by definition that subset.
# BUZSAKI_SESSIONS lists no circular session, so the two lists hold the same sessions; the
# filter matters only if one is added there.
BUZSAKI_LINEAR_SESSIONS: list[tuple[str, str]] = [
    (name, path) for name, path in BUZSAKI_SESSIONS
    if name not in BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS
]

# Sessions kept in every analysis but flagged as low-yield: the ensemble is too sparse for
# the decode window to be informative, so pooled results are reported with and without them.
# This is a reporting flag only -- it must not filter BUZSAKI_SESSIONS, eligible_sessions() or
# BUZSAKI_LINEAR_SESSIONS, or the per-session outputs that justify it would disappear.
#
# Buddy_06272013: 48 pyramidal cells at ~1 Hz, so the median 10 ms decode window carries one
# active cell and only 2% of windows carry >= 4 (other linear sessions: 19-48%). The
# likelihood surface is correspondingly flat -- the winning state beats the runner-up by
# ~0.2 nats -- and 41.5% of the emission-MAP / Viterbi mass collapses onto a single state, so
# the decode flips between the two track ends and puts spurious modes at +/- the track length
# in the jump distribution (other sessions: 8-17% single-state mass). It also yields 0 of 575
# events accepted under the traditional criteria (`replay-hmm traditional decode --dataset
# buzsaki`).
BUZSAKI_LOW_YIELD_SESSIONS = frozenset({
    "Buddy_06272013",
})

# ---------------------------------------------------------------- Pfeiffer (Pfeiffer & Foster 2015)
# Session registry: (rat, session). Linear track only -- Rat3 and Rat4 recorded only
# open-field sessions, and the eight Open* sessions are 2-D, i.e. a different pipeline.
PFEIFFER_SESSIONS: tuple[tuple[str, str], ...] = (
    ("Rat1", "Linear1"),
    ("Rat1", "Linear2"),
    ("Rat1", "Linear3"),
    ("Rat1", "Linear4"),
    ("Rat1", "Linear5"),
    ("Rat1", "Linear6"),
    ("Rat2", "Linear1"),
    ("Rat2", "Linear2"),
    ("Rat5", "Linear1"),
)

PFEIFFER_EVENT_SOURCES = ("spike-density", "swr-gated", "ripple-events", "trajectory-events")

# Tag tokens. Separators are stripped because "_" and "-" are both meaningful inside a
# "__"-delimited run-tag grammar.
PFEIFFER_EVENT_SOURCE_TAGS = {
    "spike-density": "spikedens",
    "swr-gated": "swrgated",
    "ripple-events": "ripple",
    "trajectory-events": "trajevt",
}

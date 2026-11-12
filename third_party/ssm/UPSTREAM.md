# Vendored ssm (modified fork)

This directory is a copy of [lindermanlab/ssm](https://github.com/lindermanlab/ssm) at base commit
`eb6c8aa` ("Merge pull request #173 from yahmadian/patch-1"), plus the local changes that implement the
paper's transition kernels and observation models:

- `ssm/transitions.py`: `MultiMapBehaviorStickyTransitions`, `SevenMapBehaviorStickyTransitions`,
  `MultiMapContinuousGaussianTransitions` and `SevenMapContinuousGaussianTransitions`, appended to the
  upstream file. The upstream classes are unchanged.
- `ssm/observations.py`: `FixedPoissonObservations` (Poisson rates held fixed during EM), registered as
  `fixed_poisson` in `ssm/hmm.py`.
- `ssm/messages.py`: `viterbi(pi0, Ps, ll, log_floor=LOG_EPS)`. `HMM.most_likely_states` passes
  `log_floor=0.0` when the transition model sets `hard_zero`, so exact zeros stay forbidden.
- `tests/test_multimap_continuous_gaussian.py`.

`patches/ssm_vs_eb6c8aa.patch` is the full diff against the upstream base commit. `pyproject.toml` is
added for packaging only (it declares the Cython/numpy build requirements of `ssm.cstats`).

Editing, adding or removing a `.py` file under `ssm/` changes the CV checkpoint hash
(`replay_hmm.hmm.checkpoints._cv_code_hash` keys these files as `ssm/<file>`), so CV checkpoints
written before the edit are not resumed.

Install with `pip install -e ./third_party/ssm` (a C++ compiler is required). The upstream LICENSE is kept.

Modified files:
     ssm/hmm.py                                 |    5 +
     ssm/messages.py                            |   14 +-
     ssm/observations.py                        |    4 +
     ssm/transitions.py                         | 1259 ++++++++++++++++++++++++++++
     tests/test_multimap_continuous_gaussian.py |  102 +++
     5 files changed, 1379 insertions(+), 5 deletions(-)

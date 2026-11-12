"""replay_hmm: HMM and traditional analyses of hippocampal replay trajectories.

Layering rule. The package has five layers, and each layer imports only from layers below it:

  1. `util`, `io`
  2. generic engines: `decode`, `events`, `hmm`, `jumps`, `maps`
  3. `datasets/<ds>`: loaders, geometry, and the dataset-specific adapters chosen by --dataset
  4. `analyses`, `viewer`, `plots`, `reports`
  5. `cli`, `cluster`

Generic code never imports `datasets.*`, and datasets never import each other. No library
module has import-time side effects; the one exception is `plots/style.py`, which applies the
house rcParams. `paths` sits beside layer 1: it reads the environment and pyproject.toml only.

Import cost. Every `replay-hmm` command, `--help` included, imports this file, `paths`, the
dispatcher and the command modules in `cli/`, so they import the standard library only. So do
the modules a light caller needs without the numerical stack (run-tag tokens, argparse types,
the kernel vocabulary and labels, session tables, the cluster submitter; doc/code_map.md marks
them *stdlib*): any numpy, matplotlib or ssm import they need is function-local.
"""

__version__ = "0.1.0"

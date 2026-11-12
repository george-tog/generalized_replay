"""Suite-wide fixtures."""
import os
import tempfile
from pathlib import Path

import pytest

# Headless Matplotlib for every test, set before any test module imports it. No library module
# picks the backend or the config directory at import (the CLI dispatcher does, through
# replay_hmm.util.mpl.configure_for_cli), so the tests set both here. One fixed directory under
# the system temp keeps the font cache out of the user's home. Subprocesses inherit both,
# unless a test clears them.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "replay_hmm_tests_mplconfig"))


@pytest.fixture(autouse=True, scope="session")
def _outside_a_slurm_job():
    # The --n-jobs defaults read $SLURM_CPUS_PER_TASK (replay_hmm.util.slurm), so a run inside an
    # salloc session would parse other defaults than the tests expect. The whole session runs
    # outside a job (session scope: the parse captures are cached for the session); the tests about
    # jobs set the variables themselves.
    with pytest.MonkeyPatch.context() as mp:
        for name in ("SLURM_CPUS_PER_TASK", "SLURM_ARRAY_TASK_ID"):
            mp.delenv(name, raising=False)
        yield

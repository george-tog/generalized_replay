"""`python -m replay_hmm ...` is the `replay-hmm` console script."""
import sys

from replay_hmm.cli.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

"""Shared Matplotlib color styling for every dataset's figures."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler


COLOR_PALETTE = (
    "#6278e4", # MB decode
    "#8c4bd5", # HMM decode
    "#60a8ff", # HMM generated / synthetic
    "#3dceb1", # MB traditional
    "#ec78b6", # remote replays
)



def house_rcparams() -> dict:
    """The house rcParams, as a dict for `plt.rc_context`.

    A figure drawn and saved inside `plt.rc_context(house_rcparams())` gets the house style
    whatever the global rcParams hold, so it does not depend on the import-time call below
    having run, or on nothing having changed them since (the CV strip and budget figures use
    this).
    """
    return {
        "axes.prop_cycle": cycler(color=COLOR_PALETTE),
        # Equivalent to sns.despine() defaults for every subsequently created
        # axis, including figures made inside helper functions.
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def apply_analysis_plot_style() -> None:
    """Install the shared color cycle and Seaborn-style despined axes."""
    mpl.rcParams.update(house_rcparams())


apply_analysis_plot_style()

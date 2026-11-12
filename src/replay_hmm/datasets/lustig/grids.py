"""Lustig decode-grid defaults (window width / step, seconds).

MB_DECODE_GRID:          the memoryless-Bayes jump analysis (the HMM grid).
HMM_DECODE_GRID:         the HMM fit and everything read back from its saved fits.
TRADITIONAL_DECODE_GRID: the traditional replay classifier.

`decode.config.DecodeConfig`'s own field defaults equal `HMM_DECODE_GRID`; they stay literal
there because they are hashed into the fit's cfg tag.
"""
from replay_hmm.decode.grid import DecodeGrid

HMM_DECODE_GRID = DecodeGrid(tbin_s=0.010, tbin_step_s=0.010)
MB_DECODE_GRID = HMM_DECODE_GRID
TRADITIONAL_DECODE_GRID = DecodeGrid(tbin_s=0.020, tbin_step_s=0.005)

"""Buzsaki decode-grid defaults (window width / step, seconds).

MB_DECODE_GRID:          the memoryless-Bayes jump analysis.
HMM_DECODE_GRID:         the HMM fit and everything read back from its saved fits.
TRADITIONAL_DECODE_GRID: the traditional replay classifier.
"""
from replay_hmm.decode.grid import DecodeGrid

MB_DECODE_GRID = DecodeGrid(tbin_s=0.020, tbin_step_s=0.020)
HMM_DECODE_GRID = DecodeGrid(tbin_s=0.010, tbin_step_s=0.010)
TRADITIONAL_DECODE_GRID = DecodeGrid(tbin_s=0.020, tbin_step_s=0.005)

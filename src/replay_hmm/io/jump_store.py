"""The Lustig HMM jump distribution's saved arrays: jump npz files and transition kernels.

Keys are `<model>__<rat>__<session>` (per session) and `<model>__<rat>` (pooled per rat), plus
`bounds__<rat>__<session>` for each session's map boundaries in the transition-kernel npz.
`replay-hmm jumps --dataset lustig` writes them; `--skip-process` and the event viewer read
them back.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from replay_hmm.io import names


def discover_saved_jump_sessions(results_dir: Path) -> list[tuple[str, str]]:
    metadata_path = results_dir / "jump_distribution_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        pairs = metadata.get("rat_session_pairs", [])
        if pairs:
            return [(str(rat), str(session)) for rat, session in pairs]

    jumps_path = results_dir / "jump_distributions.npz"
    if not jumps_path.exists():
        return []

    pairs = set()
    with np.load(jumps_path) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) == 3:
                _model_name, rat, session = parts
                pairs.add((rat, session))

    return sorted(pairs)


def save_transition_kernels(
    results_dir: Path,
    transition_kernels: dict[tuple[str, str, str], np.ndarray],
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for (model_name, rat, session), P in transition_kernels.items():
        arrays[f"{model_name}__{rat}__{session}"] = np.asarray(P, dtype=float)

    for (rat, session), bounds in lin_map_bounds_by_session.items():
        arrays[f"bounds__{rat}__{session}"] = np.asarray(bounds, dtype=int)

    out_path = results_dir / names.TRANSITION_KERNELS_NAME
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_saved_transition_kernels(
    results_dir: Path,
    kernels: list[str],
    session_pairs: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    filepath = results_dir / names.TRANSITION_KERNELS_NAME
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    kernel_set = set(kernels)
    requested_sessions = set(session_pairs)
    transition_kernels: dict[tuple[str, str, str], np.ndarray] = {}
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray] = {}

    with np.load(filepath) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 3:
                continue

            first, rat, session = parts
            if (rat, session) not in requested_sessions:
                continue

            if first == "bounds":
                lin_map_bounds_by_session[(rat, session)] = np.asarray(saved[key], dtype=int)
                continue

            model_name = first
            if model_name not in kernel_set:
                continue
            transition_kernels[(model_name, rat, session)] = np.asarray(saved[key], dtype=float)

    if not transition_kernels:
        raise SystemExit(f"No saved transition kernels in {filepath} matched the requested run.")

    return transition_kernels, lin_map_bounds_by_session


def load_saved_jump_distributions(
    results_dir: Path,
    kernels: list[str],
    session_pairs: list[tuple[str, str]],
    filename: str = "jump_distributions.npz",
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str, str], np.ndarray]]:
    filepath = results_dir / filename
    if not filepath.exists():
        raise SystemExit(f"Cannot --skip-process because {filepath} does not exist.")

    kernel_set = set(kernels)
    requested_sessions = set(session_pairs)
    requested_rats = {rat for rat, _session in session_pairs}

    jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]] = {}
    jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray] = {}

    with np.load(filepath) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 3:
                continue

            model_name, rat, session = parts
            if model_name not in kernel_set or (rat, session) not in requested_sessions:
                continue

            jumps = np.asarray(saved[key], dtype=int)
            jumps_by_model_rat.setdefault((model_name, rat), []).append(jumps)
            jumps_by_model_rat_session[(model_name, rat, session)] = jumps

        if jumps_by_model_rat:
            return jumps_by_model_rat, jumps_by_model_rat_session

        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 2:
                continue

            model_name, rat = parts
            if model_name not in kernel_set or rat not in requested_rats:
                continue

            jumps_by_model_rat[(model_name, rat)] = [np.asarray(saved[key], dtype=int)]

    if not jumps_by_model_rat:
        raise SystemExit(f"No saved jump arrays in {filepath} matched the requested run.")

    return jumps_by_model_rat, jumps_by_model_rat_session

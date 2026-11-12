"""Saving figures: `savefig` with retry and backoff for transient filesystem errors.

A save that keeps failing on a retryable errno, or fails on any other OSError, falls back to
/tmp/mb_jump_fig_fallback/ with a RuntimeWarning.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path


_SAVEFIG_RETRY_ERRNOS = {11, 16, 35, 60, 110}


def _savefig_with_retry(
    fig,
    output_path: Path,
    *,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
    max_attempts: int = 4,
) -> str:
    """Save matplotlib figure with retry/backoff for transient filesystem timeouts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, object] = {"dpi": int(dpi)}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches

    last_exc: Exception | None = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            fig.savefig(output_path, **save_kwargs)
            return str(output_path)
        except OSError as exc:
            last_exc = exc
            err = getattr(exc, "errno", None)
            retryable = err in _SAVEFIG_RETRY_ERRNOS
            if (not retryable) or attempt >= int(max_attempts):
                break
            time.sleep(0.4 * float(attempt))

    fallback_dir = Path("/tmp") / "mb_jump_fig_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / output_path.name
    try:
        fig.savefig(fallback_path, **save_kwargs)
        warnings.warn(
            f"savefig failed for {output_path} ({last_exc}); wrote fallback: {fallback_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return str(fallback_path)
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise

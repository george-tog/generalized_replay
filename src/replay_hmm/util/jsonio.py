"""JSON encoding for numpy scalars and arrays."""

import numpy as np


def json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")

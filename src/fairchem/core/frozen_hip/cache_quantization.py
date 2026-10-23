"""Measure FP16 cache error without slow FP16-to-FP64 conversion."""

from __future__ import annotations

import numpy as np


def quantization_statistics(source, stored):
    """Return squared error, squared reference, and maximum absolute error.

    FP32 products use FP64 sums. The source range must fit FP16. A conservative
    margin on the final relative norm covers product rounding. Tiny blocks use
    FP64 products so an underflow cannot hide a failed conversion.
    """
    maximum = float(np.max(np.abs(source), initial=0))
    if maximum < 1e-12:
        reference = source.astype(np.float64)
        difference = reference - stored.astype(np.float64)
        return (
            float(np.dot(difference, difference)),
            float(np.dot(reference, reference)),
            float(np.max(np.abs(difference), initial=0)),
        )
    difference = source - stored.astype(np.float32)
    return (
        float(np.sum(difference * difference, dtype=np.float64)),
        float(np.sum(source * source, dtype=np.float64)),
        float(np.max(np.abs(difference), initial=0)),
    )


def relative_error_bound(squared_error, squared_reference):
    """Cover FP32 products and FP64 reduction rounding conservatively."""
    relative = (squared_error / max(squared_reference, 1e-300)) ** 0.5
    return relative, relative * (1 + 2e-6)

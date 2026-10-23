"""Complete an accumulated AMP update without changing weights on overflow."""

from __future__ import annotations

import torch


def scaled_update(optimizer, scaler, parameters, *, max_norm=10.0):
    """Unscale once, clip FP32 gradients, and reject a nonfinite update.

    Return the unclipped norm and whether the optimizer applied the update.
    A rejected update resets the growth counter and halves the loss scale.
    The explicit norm check also catches overflow in the norm reduction.
    """
    parameters = list(parameters)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=max_norm, error_if_nonfinite=False
    )
    finite = bool(torch.isfinite(norm))
    if finite:
        scaler.step(optimizer)
        scaler.update()
    else:
        state = scaler.state_dict()
        state["scale"] *= 0.5
        state["_growth_tracker"] = 0
        # update clears the per-optimizer state; load restores the growth counter.
        scaler.update(new_scale=state["scale"])
        scaler.load_state_dict(state)
    return norm, finite

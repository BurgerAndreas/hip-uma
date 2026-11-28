"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Select where TensorFloat-32 applies in one HIP head training step.

TF32 in the forward pass of the head changes the predicted Hessian. The radial
MLPs (rad_func) cause nearly all of this change, and the code of the head
forward outside its child modules (projected readout) causes the rest. TF32 in
the frozen backbone and in the backward pass stays at the FP32 noise level.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

SCOPES = ("head-fp32", "scoped", "full")


def _get():
    return torch.backends.cuda.matmul.allow_tf32


def _set(enabled):
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


@contextmanager
def matmul_tf32(enabled):
    """
    Set TF32 for cuBLAS and cuDNN inside the block and restore it after.
    """
    previous = (_get(), torch.backends.cudnn.allow_tf32)
    _set(enabled)
    try:
        yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = previous


def _called_children(module):
    """
    Return the child modules with a forward of their own, through containers.
    """
    children = []
    for child in module.children():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
            children += _called_children(child)
        else:
            children.append(child)
    return children


class TF32Scope:
    """
    Apply one TF32 scope to the forward pass of a HIP head.

    full       no change: the head forward uses the process setting
    head-fp32  strict FP32 in the whole head forward
    scoped     TF32 inside the head child modules, strict FP32 in every
               rad_func module and in the code of the head forward itself

    The scope acts only while TF32 is on in the process, so the backward pass
    and the frozen backbone keep TF32 in every scope. Use `forward()` around
    each head call, including calls that activation checkpointing repeats.
    scoped switches the setting in module hooks, which compiled code does not
    run, so it needs an uncompiled head.
    """

    def __init__(self, head, scope="head-fp32"):
        if scope not in SCOPES:
            raise ValueError(f"Unknown TF32 scope: {scope}")
        self.scope = scope
        self.active = False
        self.stack = []
        self.tf32_modules = self.fp32_modules = 0
        if scope != "scoped":
            return
        on = _called_children(head)
        off = [m for name, m in head.named_modules() if name.endswith("rad_func")]
        for modules, state in ((on, True), (off, False)):
            for module in modules:
                module.register_forward_pre_hook(self._enter(state))
                module.register_forward_hook(self._exit, always_call=True)
        self.tf32_modules, self.fp32_modules = len(on), len(off)

    def _enter(self, state):
        def hook(_module, _inputs):
            if self.active:
                self.stack.append(_get())
                _set(state)

        return hook

    def _exit(self, _module, _inputs, _output):
        if self.active and self.stack:
            _set(self.stack.pop())

    @contextmanager
    def forward(self, strict=False):
        """
        Set the scope for one head forward call.

        strict gives the FP32 head forward of head-fp32 also for scoped, for
        evaluation. It does not change full.
        """
        if self.scope == "full" or not _get():
            yield
            return
        self.stack.clear()
        self.active = self.scope == "scoped" and not strict
        try:
            with matmul_tf32(False):
                yield
        finally:
            self.active = False
            self.stack.clear()

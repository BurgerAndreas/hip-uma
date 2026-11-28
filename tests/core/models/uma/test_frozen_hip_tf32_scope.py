"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import sys

import pytest
import torch

from torch import nn

from fairchem.core.frozen_hip import train
from fairchem.core.frozen_hip.tf32 import TF32Scope, matmul_tf32


def flags():
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("inner", [False, True])
def test_matmul_tf32_sets_and_restores(outer, inner):
    saved = flags()
    try:
        torch.backends.cuda.matmul.allow_tf32 = outer
        torch.backends.cudnn.allow_tf32 = outer
        with matmul_tf32(inner):
            assert flags() == (inner, inner)
        assert flags() == (outer, outer)
        with pytest.raises(RuntimeError), matmul_tf32(inner):
            raise RuntimeError
        assert flags() == (outer, outer)
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = saved


def test_tf32_scope_default_keeps_head_forward_fp32(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--data", str(tmp_path), "--output", str(tmp_path), "--tf32"],
    )
    try:
        args = train.parse_args()
    except SystemExit:
        pytest.skip("parser needs more required arguments")
    assert args.tf32
    assert args.tf32_scope == "head-fp32"


class _Record(nn.Module):
    def __init__(self, log, name):
        super().__init__()
        self.log, self.name = log, name

    def forward(self, x):
        self.log[self.name] = flags()[0]
        return x


class _Conv(nn.Module):
    def __init__(self, log, name):
        super().__init__()
        self.rad_func = _Record(log, f"{name}.rad_func")
        self.other = _Record(log, f"{name}.other")
        self.log, self.name = log, name

    def forward(self, x):
        x = self.other(self.rad_func(x))
        self.log[f"{self.name}.after_rad_func"] = flags()[0]
        return x


class _Head(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.blocks = nn.ModuleList([_Conv(log, "block0"), _Conv(log, "block1")])
        self.log = log

    def forward(self, x):
        self.log["head.before"] = flags()[0]
        for block in self.blocks:
            x = block(x)
        self.log["head.after"] = flags()[0]
        return x


@pytest.mark.parametrize(
    ("scope", "process", "strict", "expected"),
    [
        ("full", True, False, (True, True, True)),
        ("head-fp32", True, False, (False, False, False)),
        ("scoped", True, False, (False, True, False)),
        ("scoped", True, True, (False, False, False)),
        ("scoped", False, False, (False, False, False)),
        ("head-fp32", False, False, (False, False, False)),
    ],
)
def test_tf32_scope_regions(scope, process, strict, expected):
    saved = flags()
    log = {}
    head = _Head(log)
    region = TF32Scope(head, scope)
    try:
        torch.backends.cuda.matmul.allow_tf32 = process
        torch.backends.cudnn.allow_tf32 = process
        with region.forward(strict=strict):
            head(torch.zeros(1))
        own, module, radial = expected
        assert log["head.before"] == log["head.after"] == own
        for block in ("block0", "block1"):
            assert log[f"{block}.other"] == module
            assert log[f"{block}.after_rad_func"] == module
            assert log[f"{block}.rad_func"] == radial
        assert flags() == (process, process)
        assert not region.active
        assert not region.stack
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = saved


def test_tf32_scope_restores_after_error():
    saved = flags()
    region = TF32Scope(nn.Linear(1, 1), "scoped")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError), region.forward():
            raise RuntimeError
        assert flags() == (True, True)
        assert not region.active
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = saved

"""Check optional compiler-cache compatibility, ownership, and corruption handling."""

from __future__ import annotations

import pytest
import torch

from fairchem.core.frozen_hip import compiler_cache


@pytest.fixture
def fake_cache(monkeypatch):
    monkeypatch.setattr(
        compiler_cache,
        "runtime_identity",
        lambda namespace: {"namespace": namespace, "runtime": "test"},
    )
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"compiled-artifacts", None)
    )
    loaded = []
    monkeypatch.setattr(
        torch.compiler,
        "load_cache_artifacts",
        lambda value: loaded.append(value) or object(),
    )
    return loaded


def test_cache_roundtrip_and_compatibility(tmp_path, fake_cache):
    assert not compiler_cache.load_compiler_cache(tmp_path, "first")
    assert compiler_cache.save_compiler_cache(tmp_path, "first")
    assert not compiler_cache.load_compiler_cache(tmp_path, "second")
    assert compiler_cache.load_compiler_cache(tmp_path, "first")
    assert fake_cache == [b"compiled-artifacts"]
    artifact = next(tmp_path.glob("*/artifacts.bin"))
    assert artifact.stat().st_mode & 0o077 == 0


def test_corruption_never_reaches_deserialization(tmp_path, fake_cache):
    assert compiler_cache.save_compiler_cache(tmp_path, "first")
    next(tmp_path.glob("*/artifacts.bin")).write_bytes(b"incomplete")
    assert not compiler_cache.load_compiler_cache(tmp_path, "first")
    assert fake_cache == []


def test_writable_shared_artifact_is_a_cache_miss(tmp_path, fake_cache):
    assert compiler_cache.save_compiler_cache(tmp_path, "first")
    next(tmp_path.glob("*/artifacts.bin")).chmod(0o666)
    assert not compiler_cache.load_compiler_cache(tmp_path, "first")
    assert fake_cache == []


def test_cache_backend_failures_leave_compilation_available(
    tmp_path, fake_cache, monkeypatch
):
    assert compiler_cache.save_compiler_cache(tmp_path, "first")

    def fail(*args):
        raise RuntimeError("unsupported artifact")

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", fail)
    assert not compiler_cache.load_compiler_cache(tmp_path, "first")
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", fail)
    assert not compiler_cache.save_compiler_cache(tmp_path, "first")

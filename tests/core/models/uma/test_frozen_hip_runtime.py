"""Check local runtime publication without installing the scientific runtime."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / "tools/stage_frozen_hip_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_stage", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "original"
    (original / "bin").mkdir(parents=True)
    (original / "bin/python3.12").write_text("test interpreter")
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    env = tmp_path / "build/env"
    (env / "bin").mkdir(parents=True)
    (env / "bin/python").symlink_to(alias / "bin/python3.12")
    (env / "bin/python3").symlink_to("python")
    (env / "pyvenv.cfg").write_text(f"home = {alias}/bin\n")
    archive = tmp_path / "environment.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(env, arcname="env")
    destination = tmp_path / "local"
    real_run = subprocess.run
    calls = []

    def probe(command, **kwargs):
        if command[0] == "tar":
            return real_run(command, **kwargs)
        calls.append(command)
        python = Path(command[0])
        assert python.resolve() == destination / "python/bin/python3.12"
        assert (destination / "env/pyvenv.cfg").read_text() == (
            f"home = {destination}/python/bin\n"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", probe)
    return module, archive, destination, calls


def test_runtime_relocates_interpreter_and_reuses_publication(runtime):
    module, archive, destination, calls = runtime
    python = module.stage_runtime(archive, destination)
    assert python == destination / "env/bin/python"
    assert len(calls) == 1
    assert (destination / "env/bin/python3").resolve() == python.resolve()
    assert module.stage_runtime(archive, destination) == python
    assert len(calls) == 1
    ready = json.loads((destination / "runtime-ready.json").read_text())
    assert ready["mtime_ns"] == archive.stat().st_mtime_ns
    assert not list(destination.glob("runtime-stage-*"))


def test_runtime_refuses_a_different_archive_identity(runtime, tmp_path):
    module, archive, destination, _ = runtime
    module.stage_runtime(archive, destination)
    other = tmp_path / "other.tar"
    other.write_bytes(archive.read_bytes())
    with pytest.raises(ValueError, match="new destination"):
        module.stage_runtime(other, destination)


def test_failed_runtime_probe_cleans_its_own_publication(runtime, monkeypatch):
    module, archive, destination, _ = runtime
    real = module.subprocess.run

    def fail(command, **kwargs):
        if command[0] == "tar":
            return real(command, **kwargs)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.stage_runtime(archive, destination)
    assert not (destination / "env").exists()
    assert not (destination / "python").exists()
    assert not (destination / "runtime-ready.json").exists()
    assert not list(destination.glob("runtime-stage-*"))

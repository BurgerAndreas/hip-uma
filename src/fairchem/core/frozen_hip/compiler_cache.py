"""Persist supported compiler artifacts between bounded training allocations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import torch


def runtime_identity(namespace):
    import triton

    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return dict(
        version=1,
        namespace=namespace,
        source_sha256=source.hexdigest(),
        python=platform.python_version(),
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )


def _directory(root, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(root) / key


def load_compiler_cache(root, namespace):
    """Load an owned, compatible artifact; a cache miss keeps normal compilation."""
    if not hasattr(torch.compiler, "load_cache_artifacts"):
        return False
    try:
        identity = runtime_identity(namespace)
        directory = _directory(root, identity)
        metadata = directory / "manifest.json"
        record = json.loads(metadata.read_text())
        path = directory / "artifacts.bin"
        if record["identity"] != identity:
            return False
        for item in (metadata, path):
            info = item.stat()
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                return False
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            return False
        return torch.compiler.load_cache_artifacts(payload) is not None
    except Exception:
        # These artifacts are optional. Corruption or an unsupported backend
        # must never stop training or weaken compiler guards.
        return False


def save_compiler_cache(root, namespace):
    """Atomically publish artifacts without saving weights or optimizer state."""
    if not hasattr(torch.compiler, "save_cache_artifacts"):
        return False
    try:
        import fcntl

        identity = runtime_identity(namespace)
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return False
        payload, _ = result
        directory = _directory(root, identity)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (directory / "write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            temporary = directory / f"artifacts.{os.getpid()}.tmp"
            with temporary.open("wb") as output:
                temporary.chmod(0o600)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            record = dict(identity=identity, sha256=hashlib.sha256(payload).hexdigest())
            manifest = directory / f"manifest.{os.getpid()}.tmp"
            manifest.write_text(json.dumps(record, sort_keys=True) + "\n")
            manifest.chmod(0o600)
            temporary.replace(directory / "artifacts.bin")
            manifest.replace(directory / "manifest.json")
        return True
    except Exception:
        return False

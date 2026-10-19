"""Stage a training environment and its interpreter on local storage."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def stage_runtime(archive, destination):
    """Publish a local environment once, including a relocated base interpreter."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stat = archive.stat()
    identity = dict(archive=str(archive), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    with (destination / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = destination / "runtime-ready.json"
        if ready.exists():
            if json.loads(ready.read_text()) != identity:
                raise ValueError(
                    "Use a new destination for a different runtime archive"
                )
            return destination / "env/bin/python"
        if (destination / "env").exists() or (destination / "python").exists():
            raise ValueError("Destination contains an unpublished runtime")
        if shutil.disk_usage(destination).free < archive.stat().st_size * 2:
            raise ValueError(
                "Local storage cannot hold runtime staging and publication"
            )
        temporary = Path(tempfile.mkdtemp(prefix="runtime-stage-", dir=destination))
        try:
            subprocess.run(
                ["tar", "-xf", str(archive), "-C", str(temporary)], check=True
            )
            config = temporary / "env/pyvenv.cfg"
            text = config.read_text()
            entries = dict(
                line.split("=", 1) for line in text.splitlines() if "=" in line
            )
            entries = {key.strip(): value.strip() for key, value in entries.items()}
            original = Path(entries["home"]).parent
            executable = (temporary / "env/bin/python").resolve()
            if not executable.is_file():
                raise ValueError("The archived interpreter is unavailable")
            shutil.copytree(original, temporary / "python", symlinks=True)
            local = destination / "python"
            config.write_text(
                text.replace(str(original), str(local)).replace(
                    str(original.resolve()), str(local)
                )
            )
            for name in (temporary / "env/bin").glob("python*"):
                if name.is_symlink():
                    name.unlink()
                    name.symlink_to(local / "bin" / executable.name)
            (temporary / "python").rename(local)
            (temporary / "env").rename(destination / "env")
            python = destination / "env/bin/python"
            try:
                subprocess.run(
                    [
                        str(python),
                        "-c",
                        "import pathlib,sys; "
                        "assert sys.prefix != sys.base_prefix; "
                        "assert pathlib.Path(sys.base_prefix).resolve() == "
                        "pathlib.Path(sys.argv[1]).resolve()",
                        str(local),
                    ],
                    check=True,
                )
            except BaseException:
                shutil.rmtree(destination / "env")
                shutil.rmtree(local)
                raise
            ready.write_text(json.dumps(identity, sort_keys=True) + "\n")
            return python
        finally:
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(stage_runtime(args.archive, args.destination))


if __name__ == "__main__":
    main()

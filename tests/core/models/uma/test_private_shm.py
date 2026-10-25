"""Exercise real spawn workers and signal delivery in a private Linux mount."""

import json
import os
import selectors
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[4] / "scripts/with_private_shm.sh"


def line(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(30), "Private-memory probe did not respond"
    value = process.stdout.readline()
    assert value, process.stderr.read()
    return json.loads(value)


def available():
    if sys.platform != "linux":
        pytest.skip("Linux namespaces are required")
    result = subprocess.run(
        ["bash", str(SCRIPT), "true"], capture_output=True, text=True, timeout=10
    )
    if result.returncode:
        pytest.skip("This environment does not permit user/mount namespaces")


def test_spawn_workers_survive_host_cleanup(tmp_path):
    available()
    marker = "hip-test-" + uuid.uuid4().hex
    host_marker = Path("/dev/shm") / marker
    host_marker.write_text("host")
    probe = tmp_path / "probe.py"
    probe.write_text('''
import json, multiprocessing as mp, os, sys
from pathlib import Path

def worker(index, event, output):
    event.wait(10)
    output.put((index, sum(range(1000))))

if __name__ == "__main__":
    marker = Path("/dev/shm") / sys.argv[1]
    assert not marker.exists()
    marker.write_text("private")
    ctx = mp.get_context("spawn")
    event, output = ctx.Event(), ctx.Queue()
    name = event._cond._lock._semlock.name
    assert (Path("/dev/shm") / ("sem." + name.lstrip("/"))).exists()
    print(json.dumps(dict(pid=os.getpid(), semaphore=name)), flush=True)
    assert sys.stdin.readline().strip() == "continue"
    workers = [ctx.Process(target=worker, args=(i,event,output)) for i in range(2)]
    for p in workers: p.start()
    event.set()
    results = sorted(output.get(timeout=20) for _ in workers)
    for p in workers:
        p.join(20)
        assert p.exitcode == 0
    output.close()
    output.join_thread()
    assert marker.read_text() == "private"
    print(json.dumps(dict(results=results)), flush=True)
''')
    process = subprocess.Popen(
        ["bash", str(SCRIPT), sys.executable, str(probe), marker],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = line(process)
        assert ready["pid"] == process.pid
        name = "sem." + ready["semaphore"].lstrip("/")
        assert not (Path("/dev/shm") / name).exists()
        # Remove only this test's host object. The private object remains intact.
        host_marker.unlink()
        process.stdin.write("continue\n")
        process.stdin.flush()
        assert line(process)["results"] == [[0, 499500], [1, 499500]]
        assert process.wait(timeout=30) == 0, process.stderr.read()
        assert not host_marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        host_marker.unlink(missing_ok=True)


def test_checkpoint_signal_reaches_unchanged_trainer_pid():
    available()
    code = (
        "import os,signal,json; "
        "signal.signal(signal.SIGUSR1,lambda *_:exit(77)); "
        "print(json.dumps(dict(pid=os.getpid())),flush=True); signal.pause()"
    )
    process = subprocess.Popen(
        ["bash", str(SCRIPT), sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert line(process)["pid"] == process.pid
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=10) == 77
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.parametrize("size", ["0", "-1G", "1G,mode=0777", "1G;true"])
def test_invalid_shared_memory_size_is_rejected(size):
    result = subprocess.run(
        ["bash", str(SCRIPT), "true"],
        env={**os.environ, "HIP_SHM_SIZE": size},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "positive size" in result.stderr

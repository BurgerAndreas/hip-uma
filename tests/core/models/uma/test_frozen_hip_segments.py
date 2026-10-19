"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Exercise allocation continuation without a GPU or scheduler service.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/train_frozen_hip_segment.sh"


@pytest.fixture()
def allocation(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    code = tmp_path / "code"
    code.mkdir()

    def executable(name, source):
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
        return path

    python = executable(
        "fake_python",
        """
import json, os, signal, sys
from pathlib import Path
if sys.argv[1] == "-":
    sys.argv = sys.argv[1:]
    exec(sys.stdin.read())
    raise SystemExit
args = sys.argv[3:]
root = Path(args[args.index("--output")+1])
(root/"arguments.json").write_text(json.dumps(args))
with (root/"events.jsonl").open("a") as f: f.write(json.dumps("train")+"\\n")
def finish(*_):
    step = int(os.environ["OUTPUT_STEP"])
    if os.environ.get("STOP_DURING_TRAIN"): (root/"STOP").touch()
    (root/"session.json").write_text(json.dumps(dict(step=step, frozen_backbone_verified=True)))
    if step >= int(args[args.index("--steps")+1]):
        (root/"final_evaluation.json").write_text(json.dumps(dict(step=step, from_last_checkpoint=True)))
    raise SystemExit(int(os.environ.get("FAIL_TRAIN", "0")))
if os.environ.get("WAIT_FOR_SIGNAL"):
    signal.signal(signal.SIGUSR1, finish)
    os.write(int(os.environ["READY_FD"]), b"1")
    signal.pause()
finish()
""",
    )
    executable(
        "sbatch",
        """
import json, os, sys
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
path = root/"submissions.jsonl"
count = len(path.read_text().splitlines()) if path.exists() else 0
fail = os.environ.get("FAIL_SUBMIT") == "all" or (os.environ.get("FAIL_SUBMIT") == "first" and count == 0)
with path.open("a") as f:
    f.write(json.dumps(dict(args=sys.argv[1:], segment=os.environ["SEGMENT_INDEX"], failed=fail))+"\\n")
with (root/"events.jsonl").open("a") as f:
    f.write(json.dumps("submit_failed" if fail else "submit_ok")+"\\n")
if fail: raise SystemExit(1)
print(5555 + count)
""",
    )
    executable(
        "scancel",
        """
import json, os, sys
from pathlib import Path
Path(os.environ["RUN_ROOT"], "cancelled.json").write_text(json.dumps(sys.argv[1:]))
""",
    )
    executable(
        "flock", 'import os, sys\nsys.exit(int(os.environ.get("LOCK_FAIL", "0")))\n'
    )
    executable("timeout", "import os, sys\nos.execvp(sys.argv[2], sys.argv[2:])\n")
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "PYTHON_BIN": str(python),
        "CODE_ROOT": str(code),
        "DATA_ROOT": str(tmp_path / "data"),
        "PACKED_ROOT": str(tmp_path / "packed"),
        "RUN_ROOT": str(run),
        "LOSS_NAME": "mae",
        "TARGET_STEPS": "2000000",
        "OUTPUT_STEP": "100",
        "CONTINUATION_SCRIPT": str(tmp_path / "submit"),
        "SLURM_JOB_ID": "1234",
        "SEGMENT_INDEX": "0",
    }
    return run, env


def invoke(env):
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_successor_is_queued_before_training_and_resumes_exact_target(allocation):
    run, env = allocation
    result = invoke(env)
    assert result.returncode == 0, result.stderr
    args = json.loads((run / "arguments.json").read_text())
    assert args[args.index("--steps") + 1] == "2000000"
    assert args[args.index("--loss") + 1] == "mae"
    assert args[args.index("--validate-every") + 1] == "400000"
    assert args[args.index("--weight-decay") + 1] == "0"
    assert read_lines(run / "events.jsonl") == ["submit_ok", "train"]
    submission = read_lines(run / "submissions.jsonl")[0]
    assert submission["segment"] == "1"
    assert "--dependency=afterok:1234" in submission["args"]
    assert "--kill-on-invalid-dep=yes" in submission["args"]
    assert (run / "next_job_id").read_text().strip() == "5555"


def test_zero_weight_decay_reaches_trainer_and_successor(allocation):
    run, env = allocation
    result = invoke(
        {
            **env,
            "WEIGHT_DECAY": "0",
            "TARGET_STEPS": "4000000",
            "LOSS_NAME": "mass-block-rmse",
            "VALIDATE_EVERY": "400000",
        }
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((run / "arguments.json").read_text())
    assert args[args.index("--weight-decay") + 1] == "0"
    assert args[args.index("--steps") + 1] == "4000000"
    assert args[args.index("--loss") + 1] == "mass-block-rmse"
    assert args[args.index("--validate-every") + 1] == "400000"
    submission = read_lines(run / "submissions.jsonl")[0]
    assert "--export=ALL" in submission["args"]
    assert submission["segment"] == "1"


def test_completion_cancels_successor_and_future_allocations_skip_training(allocation):
    run, env = allocation
    result = invoke({**env, "OUTPUT_STEP": "2000000"})
    assert result.returncode == 0, result.stderr
    assert json.loads((run / "cancelled.json").read_text()) == ["5555"]
    events = read_lines(run / "events.jsonl")
    result = invoke(env)
    assert result.returncode == 0, result.stderr
    assert read_lines(run / "events.jsonl") == events


def test_scheduler_failure_retries_only_after_computation(allocation):
    run, env = allocation
    result = invoke({**env, "FAIL_SUBMIT": "first"})
    assert result.returncode == 0, result.stderr
    assert read_lines(run / "events.jsonl") == ["submit_failed", "train", "submit_ok"]


def test_scheduler_outage_preserves_checkpoint_and_reports_failure(allocation):
    run, env = allocation
    result = invoke({**env, "FAIL_SUBMIT": "all"})
    assert result.returncode != 0
    assert json.loads((run / "session.json").read_text())["step"] == 100
    assert read_lines(run / "events.jsonl") == [
        "submit_failed",
        "train",
        "submit_failed",
    ]


def test_training_failure_does_not_submit_another_successor(allocation):
    run, env = allocation
    result = invoke({**env, "FAIL_TRAIN": "7"})
    assert result.returncode == 7
    assert read_lines(run / "events.jsonl") == ["submit_ok", "train"]


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGTERM])
def test_signals_preserve_progress_without_blocking_scheduler_restarts(allocation, sig):
    run, env = allocation
    reader, writer = os.pipe()
    if writer == 9:
        replacement = os.dup(writer)
        os.close(writer)
        writer = replacement
    child = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env={**env, "WAIT_FOR_SIGNAL": "1", "READY_FD": str(writer)},
        pass_fds=(writer,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(writer)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(reader, selectors.EVENT_READ)
            if not selector.select(timeout=10):
                child.kill()
                pytest.fail("The test trainer did not become ready")
            assert os.read(reader, 1) == b"1"
    finally:
        os.close(reader)
    child.send_signal(sig)
    stdout, stderr = child.communicate(timeout=10)
    assert child.returncode == 0, (stdout, stderr)
    assert json.loads((run / "session.json").read_text())["step"] == 100
    assert not (run / "STOP").exists()
    assert not (run / "cancelled.json").exists()


def test_repeated_allocation_start_reuses_its_accepted_successor(allocation):
    run, env = allocation
    first = invoke(env)
    assert first.returncode == 0, first.stderr
    second = invoke({**env, "OUTPUT_STEP": "200"})
    assert second.returncode == 0, second.stderr
    assert len(read_lines(run / "submissions.jsonl")) == 1
    assert "Reusing continuation" in second.stdout
    assert json.loads((run / "session.json").read_text())["step"] == 200


def test_scheduler_requeue_replaces_an_earlier_failed_dependency(allocation):
    run, env = allocation
    assert invoke(env).returncode == 0
    second = invoke({**env, "SLURM_RESTART_COUNT": "1"})
    assert second.returncode == 0, second.stderr
    assert len(read_lines(run / "submissions.jsonl")) == 2
    assert json.loads((run / "cancelled.json").read_text()) == ["5555"]
    assert (run / "next_job_id").read_text().strip() == "5556"
    third = invoke({**env, "SLURM_RESTART_COUNT": "1"})
    assert third.returncode == 0, third.stderr
    assert len(read_lines(run / "submissions.jsonl")) == 2


def test_explicit_stop_file_prevents_training_and_new_allocations(allocation):
    run, env = allocation
    (run / "STOP").touch()
    result = invoke(env)
    assert result.returncode == 0, result.stderr
    assert not (run / "events.jsonl").exists()


def test_explicit_stop_during_training_saves_then_cancels_successor(allocation):
    run, env = allocation
    result = invoke({**env, "STOP_DURING_TRAIN": "1"})
    assert result.returncode == 0, result.stderr
    assert json.loads((run / "session.json").read_text())["step"] == 100
    assert json.loads((run / "cancelled.json").read_text()) == ["5555"]


def test_failed_allocation_does_not_reuse_an_invalid_dependency(allocation):
    run, env = allocation
    first = invoke({**env, "FAIL_TRAIN": "7"})
    assert first.returncode == 7
    assert not (run / "continuation.record").exists()
    second = invoke(env)
    assert second.returncode == 0, second.stderr
    assert len(read_lines(run / "submissions.jsonl")) == 2

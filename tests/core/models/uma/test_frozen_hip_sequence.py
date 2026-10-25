"""Compare real prefetched data streams across interruptions and worker counts."""

from __future__ import annotations

import copy
import itertools
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.packed import (
    PackedHessianStream,
    open_packed,
    write_packed,
)
from fairchem.core.frozen_hip.sequence import (
    PackedSequence,
    PlannedPackedDataset,
    capture_rng,
    fixed_rng,
    pin_manifest,
    pin_validation_order,
    restore_rng,
    seed_rng,
    seed_update,
)


@pytest.fixture
def bundles(tmp_path):
    rng = np.random.default_rng(41)
    for file in range(4):
        sizes = np.array([2, 5, 3, 4, 2, 7, 3], dtype=np.int64)
        atoms = int(sizes.sum())
        pairs = np.r_[0, np.cumsum(sizes * (sizes - 1) * 9 // 2)]
        arrays = dict(
            natoms=sizes,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            atomic_numbers=np.ones(atoms, dtype=np.int64),
            coords=rng.normal(size=(atoms, 3)),
            charge=np.zeros(len(sizes), dtype=np.int64),
            spin=np.ones(len(sizes), dtype=np.int64),
            omol_index=np.arange(file * 100 + 1, file * 100 + 1 + len(sizes)),
            pair_ptr=pairs,
            pair_targets=rng.normal(size=pairs[-1]).astype(np.float32),
            diagonal_targets=rng.normal(size=atoms * 6).astype(np.float32),
            asymmetry_mse=np.zeros(len(sizes), dtype=np.float32),
            features=rng.normal(size=(atoms, 9, 4)).astype(np.float32),
        )
        write_packed(
            tmp_path / f"bundle_{file:03d}.hip",
            arrays,
            dict(
                training_split="release-v2",
                backbone_sha256="frozen",
                backbone_precision="fp32",
                tf32=True,
                backbone_graph_version=2,
                fused_backbone=False,
                grouped_mole=False,
                feature_scale=1.0,
            ),
        )
    return tmp_path


def stream(root, **kw):
    return PackedHessianStream(
        root,
        markers=[p.name for p in sorted(root.glob("*.hip"))],
        backbone_sha256="frozen",
        max_atoms=14,
        max_batch=3,
        window_files=2,
        pair_budget=kw.pop("pair_budget", 50),
        fast_packing=kw.pop("fast_packing", True),
        include_cache_metadata=False,
        **kw,
    )


def loader(s, sequence, workers):
    return DataLoader(
        PlannedPackedDataset(s),
        sampler=sequence,
        batch_size=None,
        num_workers=workers,
        generator=torch.Generator().manual_seed(7),
        **(dict(multiprocessing_context="spawn", prefetch_factor=4) if workers else {}),
    )


def close(iterator):
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def ids(batch):
    return [int(sid) for micro in batch for sid in micro.sid]


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("cut", [1, 7, 19])
def test_restart_exact_sequence_with_prefetch(bundles, workers, cut):
    s = stream(bundles)
    reference = iter(loader(s, PackedSequence(s), 0))
    expected = [ids(next(reference)) for _ in range(35)]
    s = stream(bundles)
    plan = PackedSequence(s)
    it = iter(loader(s, plan, workers))
    prefix = []
    for _ in range(cut):
        batch = next(it)
        prefix.append(ids(batch))
    # Workers have prefetched beyond this cursor. Only consumed data count.
    saved = plan.state(batch[0].hip_sequence_after, cut, "prefix")
    close(it)
    s = stream(bundles)
    restored = PackedSequence(s)
    restored.validate_state(saved)
    it = iter(loader(s, restored, 2 if workers == 0 else 0))
    actual = prefix + [ids(next(it)) for _ in range(35 - cut)]
    close(it)
    assert actual == expected


def test_epochs_have_no_duplicates_or_omissions(bundles):
    s = stream(bundles)
    plan = PackedSequence(s)
    dataset = PlannedPackedDataset(s)
    epochs = [[], []]
    for request in plan:
        if request.before[0] >= 2:
            break
        epochs[request.before[0]].extend(ids(dataset[request]))
    expected = sorted(int(x) for m in plan.metadata for x in m["ids"])
    for epoch in epochs:
        assert sorted(epoch) == expected
    assert epochs[0] != epochs[1]


@pytest.mark.parametrize("fast", [False, True])
def test_physical_caps_preserve_exact_sample_order(bundles, fast):
    actual = []
    for cap in [20, 100000]:
        s = stream(bundles, pair_budget=cap, fast_packing=fast)
        p = PackedSequence(s)
        d = PlannedPackedDataset(s)
        actual.append([ids(d[r]) for r in itertools.islice(p, 30)])
    assert actual[0] == actual[1]


def test_logical_batch_caps_only_regroup_the_canonical_sample_stream(bundles):
    results = []
    for max_batch, max_atoms in [(3, 14), (6, 28)]:
        s = stream(bundles)
        s.max_batch = max_batch
        s.max_atoms = max_atoms
        p = PackedSequence(s)
        d = PlannedPackedDataset(s)
        seen = []
        for request in p:
            if request.before[0]:
                break
            seen.extend(ids(d[request]))
        results.append(seen)
    assert results[0] == results[1]


def test_model_rng_does_not_choose_data_order(bundles):
    orders = []
    for seed in [2, 99]:
        seed_rng(seed)
        torch.rand(1234)
        s = stream(bundles)
        p = PackedSequence(s)
        orders.append([r.references for r in itertools.islice(p, 20)])
    assert orders[0] == orders[1]


def test_checkpoint_rejects_changed_data_or_logical_batching(bundles):
    s = stream(bundles)
    p = PackedSequence(s)
    saved = p.state((0, 0, 0), 0, "prefix")
    other = stream(bundles)
    other.max_batch = 6
    with pytest.raises(ValueError, match="changed"):
        PackedSequence(other).validate_state(saved)
    path = bundles / "bundle_000.hip"
    v, h = open_packed(path)
    a = {k: np.asarray(x).copy() for k, x in v.items()}
    a["charge"][0] = 2
    write_packed(
        path, a, {k: x for k, x in h.items() if k not in ["version", "arrays"]}
    )
    with pytest.raises(ValueError, match="changed"):
        PackedSequence(stream(bundles)).validate_state(saved)
    with pytest.raises(ValueError, match="changed"):
        PlannedPackedDataset(s)[next(r for r in p if 0 in r.sources)]


def test_legacy_seek_matches_uninterrupted_plan(bundles):
    s = stream(bundles)
    p = PackedSequence(s)
    requests = list(itertools.islice(p, 45))
    q = PackedSequence(s)
    q.seek_updates(31)
    assert list(itertools.islice(q, 14)) == requests[31:]


def test_training_and_validation_rng_resume_exactly(bundles):
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1)
        )

    torch.manual_seed(3)
    initial = copy.deepcopy(make_model().state_dict())

    def run(model, opt, it, start, end):
        seen = []
        for _index in range(start, end):
            batch = next(it)
            seen.append(ids(batch))
            opt.zero_grad()
            for data in batch:
                x = data.hip_features[:, 0, :].mean(0)
                loss = (model(x) - random.random() - float(np.random.random())) ** 2
                loss.backward()
            opt.step()
        return seen, batch

    def setup(workers, state=None):
        m = make_model()
        m.load_state_dict(state["model"] if state else initial)
        o = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0)
        if state:
            o.load_state_dict(state["optimizer"])
        s = stream(bundles)
        p = PackedSequence(s)
        if state:
            p.validate_state(state["sequence"])
        it = iter(loader(s, p, workers))
        restore_rng(state["rng"]) if state else seed_rng(7)
        return m, o, p, it

    a, oa, _pa, ia = setup(0)
    expected, _ = run(a, oa, ia, 0, 15)
    b, ob, pb, ib = setup(2)
    prefix, batch = run(b, ob, ib, 0, 6)
    state = dict(
        model=copy.deepcopy(b.state_dict()),
        optimizer=copy.deepcopy(ob.state_dict()),
        sequence=pb.state(batch[0].hip_sequence_after, 6, "prefix"),
        rng=capture_rng(),
    )
    close(ib)
    c, oc, _pc, ic = setup(0, state)
    # Validation and an evaluation failure cannot consume training RNG.
    with pytest.raises(RuntimeError), fixed_rng(7):
        random.random()
        np.random.random()
        torch.rand(10)
        raise RuntimeError("evaluation failure")
    suffix, _ = run(c, oc, ic, 6, 15)
    assert prefix + suffix == expected
    for x, y in zip(a.parameters(), c.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_fixed_validation_contract_and_order(tmp_path):
    m = dict(
        selection="test",
        dataset_revision="revision",
        samples=2,
        sample_ids_sha256="ids",
        entries=[{"rows": [0, 1]}],
    )
    pin_manifest(tmp_path, "monitor", m)
    pin_manifest(tmp_path, "monitor", m)
    changed = copy.deepcopy(m)
    changed["entries"][0]["rows"] = [1, 0]
    with pytest.raises(ValueError, match="changed"):
        pin_manifest(tmp_path, "monitor", changed)
    pin_validation_order(tmp_path, "monitor", [1, 2])
    with pytest.raises(ValueError, match="changed"):
        pin_validation_order(tmp_path, "monitor", [2, 1])
    partial = {**m, "complete": False, "entries": []}
    pin_manifest(tmp_path, "final", partial)
    pin_manifest(tmp_path, "final", {**m, "complete": True})
    with pytest.raises(ValueError, match="changed"):
        pin_manifest(tmp_path, "final", partial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_rng_survives_checkpoint_roundtrip(tmp_path):
    seed_rng(17)
    state = capture_rng("cuda")
    path = tmp_path / "state.pt"
    torch.save(state, path)
    expected = torch.rand(31, device="cuda")
    restored = torch.load(path, map_location="cuda", weights_only=False)
    restore_rng(restored, "cuda")
    torch.testing.assert_close(torch.rand(31, device="cuda"), expected, rtol=0, atol=0)


def test_update_random_views_do_not_depend_on_previous_work():
    def draws():
        return random.random(), float(np.random.random()), torch.rand(12)

    seed_update(7, 123)
    expected = draws()
    seed_rng(88)
    draws()
    with fixed_rng(7):
        draws()
    seed_update(7, 123)
    actual = draws()
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    seed_update(7, 124)
    assert draws()[0] != expected[0]

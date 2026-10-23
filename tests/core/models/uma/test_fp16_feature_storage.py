"""Require explicit precision selection and atomic cache conversion."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fairchem.core.frozen_hip.cache import cached_features
from tools.convert_frozen_hip_cache import convert_file


def make_cache(root, values):
    root.mkdir()
    path = root / "features.npy"
    np.save(path, values)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source_sha256="a" * 64,
                shape=list(values.shape),
                finite=True,
            )
        )
    )
    return path.with_suffix(".json")


def test_half_cache_requires_marked_storage_and_opt_in(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    values = np.random.default_rng(17).normal(size=(131, 9, 8)).astype(np.float32)
    marker = make_cache(source, values)
    task = (marker, source, output, 137, 0.0003)
    report = convert_file(task)
    assert report["relative_error"] < 0.0003
    np.testing.assert_array_equal(np.load(source / "features.npy"), values)
    with pytest.raises(ValueError, match="dtype"):
        cached_features(output, "features", "a" * 64)
    actual = cached_features(output, "features", "a" * 64, allow_fp16=True)
    np.testing.assert_array_equal(actual, values.astype(np.float16))
    assert convert_file(task)["reused"]
    with pytest.raises(ValueError, match="scientific data"):
        cached_features(output, "features", "b" * 64, allow_fp16=True)
    metadata = json.loads((output / "features.json").read_text())
    del metadata["storage_dtype"]
    (output / "features.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dtype"):
        cached_features(output, "features", "a" * 64, allow_fp16=True)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e9])
def test_invalid_half_cache_is_not_published(tmp_path, value):
    source, output = tmp_path / "source", tmp_path / "output"
    marker = make_cache(source, np.array([value], dtype=np.float32))
    with pytest.raises(ValueError):
        convert_file((marker, source, output, 128, 0.0003))
    assert not (output / "features.json").exists()
    assert not list(output.glob("*.partial"))


def test_float_cache_still_works_without_opt_in(tmp_path):
    values = np.arange(9, dtype=np.float32)
    make_cache(tmp_path / "source", values)
    np.testing.assert_array_equal(
        cached_features(tmp_path / "source", "features", "a" * 64), values
    )


def test_mixed_cached_features_use_explicit_packing_dtype():
    import torch

    from fairchem.core.frozen_hip.packed import pack_cached_window

    window = []
    for index, dtype in enumerate((np.float16, np.float32)):
        values = dict(
            natoms=np.array([2]),
            atom_ptr=np.array([0, 2]),
            pair_ptr=np.array([0, 9]),
            coords=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            atomic_numbers=np.array([1, 1]),
            charge=np.array([0]),
            spin=np.array([1]),
            omol_index=np.array([index]),
            features=np.full((2, 9, 4), 0.123456, dtype=dtype),
            pair_targets=np.full(9, 0.654321, dtype=np.float32),
            diagonal_targets=np.full(12, 0.234567, dtype=np.float32),
            asymmetry_mse=np.zeros(1, dtype=np.float32),
        )
        window.append((values, dict(feature_scale=1.0)))
    references = [(0, 0), (1, 0)]
    baseline = pack_cached_window(window, references)
    packed = pack_cached_window(window, references, feature_dtype=np.float16)
    assert baseline.hip_features.dtype == torch.float32
    assert packed.hip_features.dtype == torch.float16
    torch.testing.assert_close(
        packed.hip_features, baseline.hip_features.half(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed.hip_pair_targets, baseline.hip_pair_targets, rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed.hip_diagonal_targets, baseline.hip_diagonal_targets, rtol=0, atol=0
    )
    assert packed.sid == baseline.sid


@pytest.mark.parametrize("scale", [0.0, 1e-30, 1e-12, 1e-6, 1.0, 1e4])
def test_fast_error_bound_covers_fp64_reference(scale):
    from fairchem.core.frozen_hip.cache_quantization import (
        quantization_statistics,
        relative_error_bound,
    )

    source = (np.random.default_rng(17).normal(size=8192) * scale).astype(np.float32)
    stored = source.astype(np.float16)
    error, reference, maximum = quantization_statistics(source, stored)
    actual_reference = source.astype(np.float64)
    difference = actual_reference - stored.astype(np.float64)
    actual = np.linalg.norm(difference) / max(np.linalg.norm(actual_reference), 1e-150)
    nominal, upper = relative_error_bound(error, reference)
    assert actual <= upper
    np.testing.assert_allclose(nominal, actual, rtol=1e-6, atol=1e-12)
    assert maximum == np.max(np.abs(difference), initial=0)

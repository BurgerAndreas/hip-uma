#!/usr/bin/env python3
"""
Download selected FAIR Chemistry checkpoints and extract lightweight configs.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download

from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS


MODEL_NAMES = (
    "uma-s-1p2",
    "uma-s-1p1",
    "uma-m-1p1",
    "esen-sm-direct-all-omol",
    "esen-sm-conserving-all-omol",
    "esen-md-direct-all-omol",
)


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": "torch.Tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() <= 32:
        summary["values"] = tensor.detach().cpu().tolist()
    return summary


def _module_summary(module: torch.nn.Module) -> dict[str, Any]:
    state = module.state_dict()
    return {
        "_target_": f"{module.__class__.__module__}.{module.__class__.__name__}",
        "state_dict": {key: _to_config_value(value) for key, value in state.items()},
    }


def _to_config_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_config_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, torch.nn.Module):
        return _module_summary(value)
    if isinstance(value, dict):
        return {str(key): _to_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_config_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "_target_": f"{value.__class__.__module__}.{value.__class__.__name__}",
        "repr": repr(value),
    }


def _state_dict_stats(state_dict: dict[str, Any]) -> dict[str, Any]:
    tensors = {key: value for key, value in state_dict.items() if torch.is_tensor(value)}
    total_params = sum(int(value.numel()) for value in tensors.values())
    trainable_size_bytes = sum(
        int(value.numel() * value.element_size()) for value in tensors.values()
    )
    by_dtype: dict[str, int] = {}
    for tensor in tensors.values():
        dtype = str(tensor.dtype)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + int(tensor.numel())
    return {
        "num_tensors": len(tensors),
        "total_parameters": total_params,
        "tensor_bytes": trainable_size_bytes,
        "dtypes": by_dtype,
    }


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model_file(model_name: str, cache_dir: Path) -> dict[str, Any]:
    hf = _MODEL_CKPTS.checkpoints[model_name]
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=hf.repo_id,
            filename=hf.filename,
            subfolder=hf.subfolder,
            revision=hf.revision,
            cache_dir=cache_dir,
        )
    )

    refs: dict[str, Any] = {}
    for ref_name in ("atom_refs", "form_elem_refs"):
        ref_spec = getattr(hf, ref_name)
        if ref_spec is None:
            continue
        refs[ref_name] = {
            "path": hf_hub_download(
                repo_id=hf.repo_id,
                filename=ref_spec["filename"],
                subfolder=ref_spec["subfolder"],
                revision=hf.revision,
                cache_dir=cache_dir,
            ),
            **ref_spec,
        }

    return {
        "repo_id": hf.repo_id,
        "filename": hf.filename,
        "subfolder": hf.subfolder,
        "revision": hf.revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_md5": _file_md5(checkpoint_path),
        "references": refs,
    }


def inspect_model(model_name: str, cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    hf_info = _download_model_file(model_name, cache_dir)
    checkpoint_path = Path(hf_info["checkpoint_path"])

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state_dict = getattr(checkpoint, "ema_state_dict", None) or getattr(
        checkpoint, "model_state_dict", {}
    )
    config = {
        "model_name": model_name,
        "source": hf_info,
        "checkpoint_type": f"{checkpoint.__class__.__module__}.{checkpoint.__class__.__name__}",
        "state_dict_stats": _state_dict_stats(state_dict),
        "model_config": _to_config_value(getattr(checkpoint, "model_config", {})),
        "tasks_config": _to_config_value(getattr(checkpoint, "tasks_config", {})),
    }

    output_path = output_dir / f"{model_name}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, width=100)

    summary = {
        "model_name": model_name,
        "repo_id": hf_info["repo_id"],
        "filename": hf_info["filename"],
        "subfolder": hf_info["subfolder"],
        "checkpoint_path": hf_info["checkpoint_path"],
        "checkpoint_size_bytes": hf_info["checkpoint_size_bytes"],
        "checkpoint_md5": hf_info["checkpoint_md5"],
        "config_path": str(output_path),
        "checkpoint_type": config["checkpoint_type"],
        "state_dict_stats": config["state_dict_stats"],
        "model_config_top_level_keys": list(config["model_config"].keys()),
        "num_tasks": (
            len(config["tasks_config"])
            if isinstance(config["tasks_config"], list)
            else None
        ),
    }

    del checkpoint
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/fairchem")
    parser.add_argument(
        "--output-dir", default="configs/pretrained/model_configs"
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in args.models:
        print(f"Inspecting {model_name}", flush=True)
        summaries.append(inspect_model(model_name, cache_dir, output_dir))

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

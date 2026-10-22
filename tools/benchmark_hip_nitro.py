"""Benchmark experimental trainable fusions against the compiled HIP baseline.

Set TRAIN_ROOT (packed data and feature cache), BENCH_ROOT (new output folder),
and HIP_CHECKPOINT (dense HIP checkpoint). Run inside a GPU allocation. This
benchmark never saves optimizer changes to the source checkpoint.
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import os
import time
import traceback
from pathlib import Path
from types import MethodType

import torch

from fairchem.core.frozen_hip import edge_precision as edge_kernels
from fairchem.core.frozen_hip import linear as linear_kernels
from fairchem.core.frozen_hip.cache import feature_namespace
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.nitro_candidates import (
    _Gate,
    _NormSiLU,
    enable_candidates,
)
from fairchem.core.frozen_hip.packed import PackedHessianStream, block_rmse
from fairchem.core.frozen_hip.precision import configure_head_precision
from fairchem.core.frozen_hip.radial import enable_fused_radial
from fairchem.core.frozen_hip.readout import enable_projected_readout
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.models.uma.nn.activation import GateActivation


def main():
    root = Path(os.environ["TRAIN_ROOT"])
    out = Path(os.environ["BENCH_ROOT"])
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 32
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tests": [],
        "cases": {},
        "scope": "cached-feature complete AdamW updates; loader excluded",
    }

    def save():
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    def rel(a, b):
        return float(
            (a.detach().float() - b.detach().float()).norm()
            / b.detach().float().norm().clamp_min(1e-20)
        )

    def check(label, actual, expected, tol):
        e = rel(actual, expected)
        assert torch.isfinite(actual).all(), label
        assert e < tol, (label, e, tol)
        report["tests"].append({"test": label, "relative_l2": e, "tolerance": tol})
        print("CHECK", label, e, flush=True)

    def microchecks():
        for dtype in [torch.float32, torch.float16]:
            for rows, c in [(17, 96), (257, 128), (122150, 128)]:
                x = (
                    torch.randn(rows, c, device="cuda", dtype=dtype) * 3
                ).requires_grad_()
                w = torch.randn(c, device="cuda", requires_grad=True)
                b = torch.randn_like(w, requires_grad=True)
                y = torch.nn.functional.silu(
                    torch.nn.functional.layer_norm(x.float(), (c,), w, b, 1e-5).to(
                        dtype
                    )
                )
                g = torch.randn_like(y)
                ref = torch.autograd.grad(y, (x, w, b), g)
                z = _NormSiLU.apply(x, w, b, 1e-5)
                got = torch.autograd.grad(z, (x, w, b), g)
                check(
                    f"norm-output-{dtype}-{rows}-{c}",
                    z,
                    y,
                    0.002 if dtype == torch.float16 else 2e-5,
                )
                for name, a, r in zip(["dx", "dw", "db"], got, ref, strict=True):
                    check(
                        f"norm-{name}-{dtype}-{rows}-{c}",
                        a,
                        r,
                        0.004 if dtype == torch.float16 else 5e-5,
                    )
                del x, w, b, y, g, ref, z, got
            for mp in [False, True]:
                gate = GateActivation(2, 2, 128, m_prime=mp).cuda()
                x = torch.randn(
                    513, 9, 128, device="cuda", dtype=dtype, requires_grad=True
                )
                g = torch.randn(
                    513, 256, device="cuda", dtype=dtype, requires_grad=True
                )
                dy = torch.randn_like(x)
                y = gate(g, x)
                ref = torch.autograd.grad(y, (g, x), dy)
                z = _Gate.apply(g, x, gate.expand_index)
                got = torch.autograd.grad(z, (g, x), dy)
                check(
                    f"gate-output-{dtype}-{mp}",
                    z,
                    y,
                    0.002 if dtype == torch.float16 else 2e-5,
                )
                for name, a, r in zip(["dg", "dx"], got, ref, strict=True):
                    check(
                        f"gate-{name}-{dtype}-{mp}",
                        a,
                        r,
                        0.004 if dtype == torch.float16 else 2e-5,
                    )
        save()

    microchecks()
    model = FrozenHIP("cuda", backbone_graph_version=4)
    frozen = parameter_hash(model.backbone)
    base = model.head
    checkpoint = Path(os.environ["HIP_CHECKPOINT"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(state["head"])
    checkpoint_step = int(state["step"])
    del state
    initial = copy.deepcopy(base.state_dict())
    cache = (
        root
        / "features"
        / feature_namespace(
            frozen,
            tf32=True,
            backbone_precision="fp16",
            grouped_mole=True,
            backbone_graph_version=4,
            fused_backbone=True,
        )
    )
    packed = root / "data/packed-targets"
    stream = PackedHessianStream(
        packed,
        markers=sorted(p.name for p in packed.glob("*.hip")),
        backbone_sha256=frozen,
        backbone_precision="fp16",
        backbone_graph_version=4,
        fused_backbone=True,
        tf32=True,
        grouped_mole=True,
        feature_cache=cache,
        max_passes=1,
        max_batch=64,
        max_atoms=2560,
        pair_budget=524288,
        window_files=4,
        seed=73,
        include_cache_metadata=False,
        fast_packing=True,
    )
    host = []
    for logical in itertools.islice(iter(stream), 16):
        for raw in logical:
            raw.hip_sizes = raw.natoms.tolist()
            host.append(raw)
    host.sort(key=lambda d: sum(n * (n - 1) for n in d.hip_sizes))
    selected = [host[0], host[len(host) // 2], host[-1]]
    batches = []
    for raw in selected:
        sizes = raw.hip_sizes
        d = raw.to("cuda")
        d.hip_sizes = sizes
        d.hip_fused_loss = True
        batches.append(d)
        del raw
    report["checkpoint_step"] = checkpoint_step
    report["batches"] = [
        {
            "geometries": len(d.hip_sizes),
            "atoms": sum(d.hip_sizes),
            "max_atoms": max(d.hip_sizes),
            "directed_edges": sum(n * (n - 1) for n in d.hip_sizes),
        }
        for d in batches
    ]
    print("BATCHES", json.dumps(report["batches"]), flush=True)
    del host, selected, stream
    gc.collect()

    def configure(case):
        h = copy.deepcopy(base)
        h.load_state_dict(initial)
        enable_fused_rotations(h)
        enable_projected_readout(h)
        enable_fused_radial(h)
        configure_head_precision(h, "fp16-edge")
        enable_candidates(
            h, gate=case in ["gate", "both"], norm=case in ["norm", "both"]
        )
        if case in ("atomic", "atomic_wide"):
            if case == "atomic_wide":
                edge_kernels.GRID_E_STRIDE = 131072
            from fairchem.core.frozen_hip.edge_precision import MixedBackend

            for m in h.modules():
                if hasattr(m, "backend"):
                    m.backend = MixedBackend(False)
        if case in ("scaled", "scaled14"):
            if case == "scaled14":
                from fairchem.core.frozen_hip.nitro_candidates import (
                    nitro_scaled_quantize,
                )

                linear_kernels.quantize = nitro_scaled_quantize
            from fairchem.core.frozen_hip.linear import _ScaledLinear

            def scaled(self, x):
                return _ScaledLinear.apply(x.float(), self.weight, self.bias).half()

            for block in [b.edge_wise for b in h.blocks] + [h.edge_readout]:
                for m in block.modules():
                    if isinstance(m, torch.nn.Linear):
                        m.forward = MethodType(scaled, m)
        if case == "split_dw":
            from fairchem.core.frozen_hip.nitro_candidates import enable_dense_split_dw

            enable_dense_split_dw(int(os.environ.get("HIP_SPLIT_CHUNK", "2048")))
        if case == "norm_compile":
            h.norm.compile(dynamic=True)
        h.train()
        return h

    def forward(h, d):
        return h(
            d,
            {
                "node_embedding": d.hip_features.float()
                * getattr(d, "hip_feature_scale", 1.0),
                "batch": d.batch,
            },
        )

    def loss(h, d):
        p = forward(h, d)
        return block_rmse(p["edge_blocks"], p["node_blocks"], d, mass_weighted=True)

    def gradients(h, d):
        torch.manual_seed(19)
        h.zero_grad(set_to_none=True)
        p = forward(h, d)
        v = block_rmse(p["edge_blocks"], p["node_blocks"], d, mass_weighted=True)
        (v * 1024).backward()
        g = {
            n: p.grad.detach().clone() / 1024
            for n, p in h.named_parameters()
            if p.grad is not None
        }
        return {k: v.detach().clone() for k, v in p.items()}, v.detach().clone(), g

    def compile_head(h):
        for b in h.blocks:
            b.compile(dynamic=True)
        h.projected_edge_irreps = torch.compile(h.projected_edge_irreps, dynamic=True)

    reference = configure("baseline")
    eager_p, eager_l, eager_g = gradients(reference, batches[-1])
    compile_head(reference)
    refp, refl, refg = gradients(reference, batches[-1])
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    def bench(case):
        h = configure(case)
        p, v, g = gradients(h, batches[-1])
        assert set(g) == set(eager_g), "Parameter gradient set changed"
        output_error = max(rel(p[k], eager_p[k]) for k in p)
        grad_error = (
            (
                sum((g[k] - eager_g[k]).square().sum() for k in g)
                / sum(x.square().sum() for x in eager_g.values())
            )
            .sqrt()
            .item()
        )
        loss_error = rel(v, eager_l)
        finite = all(bool(torch.isfinite(x).all()) for x in g.values())
        row = {
            "output_relative_l2": output_error,
            "loss_relative": loss_error,
            "gradient_relative_l2": grad_error,
            "finite_gradients": finite,
            "parameter_gradient_count": len(g),
        }
        print("FIDELITY", case, json.dumps(row), flush=True)
        del p, v, g
        if not finite or grad_error > 0.01 or loss_error > 0.002:
            row["rejected"] = "fidelity threshold"
            return row
        h.load_state_dict(initial)
        h.zero_grad(set_to_none=True)
        compile_head(h)
        compiled_p, compiled_loss, compiled_g = gradients(h, batches[-1])
        assert set(compiled_g) == set(refg), "Compiled parameter gradient set changed"
        row["compiled_output_relative_l2"] = max(
            rel(compiled_p[k], refp[k]) for k in compiled_p
        )
        row["compiled_loss_relative"] = rel(compiled_loss, refl)
        row["compiled_gradient_relative_l2"] = (
            (
                sum((compiled_g[k] - refg[k]).square().sum() for k in compiled_g)
                / sum(x.square().sum() for x in refg.values())
            )
            .sqrt()
            .item()
        )
        if (
            row["compiled_gradient_relative_l2"] > 0.01
            or row["compiled_loss_relative"] > 0.002
            or not all(bool(torch.isfinite(x).all()) for x in compiled_g.values())
        ):
            row["rejected"] = "compiled fidelity threshold"
            return row
        del compiled_p, compiled_loss, compiled_g
        h.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(h.parameters(), lr=0.001, weight_decay=0, fused=True)
        params = list(h.parameters())

        def update(d, h=h, opt=opt, params=params):
            opt.zero_grad(set_to_none=True)
            v = loss(h, d)
            (v * 1024).backward()
            torch._foreach_div_([p.grad for p in params if p.grad is not None], 1024)
            torch.nn.utils.clip_grad_norm_(
                params, 10, error_if_nonfinite=False, foreach=True
            )
            opt.step()
            return v

        for _ in range(3):
            for d in batches:
                update(d)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _repeat in range(4):
            start = time.monotonic()
            for _ in range(8):
                for d in batches:
                    last = update(d)
            torch.cuda.synchronize()
            timings.append((time.monotonic() - start) / 24)
        row.update(
            seconds_per_update=timings,
            median_seconds=sorted(timings)[len(timings) // 2],
            samples_s=(sum(len(d.hip_sizes) for d in batches) / 3)
            / (sum(timings) / len(timings)),
            peak_GB=torch.cuda.max_memory_allocated() / 1e9,
            last_loss=float(last.detach()),
        )
        if case in ["baseline", "both"]:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for d in batches:
                    update(d)
            (out / ("operators-" + case + ".txt")).write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
            )
        del update, opt, h, params
        gc.collect()
        torch.cuda.empty_cache()
        return row

    original_radial = (edge_kernels.mixed_radial, edge_kernels.mixed_radial_grad)

    original_half_mm = edge_kernels.half_mm_float_output
    original_quantize = linear_kernels.quantize
    original_grid = edge_kernels.GRID_E_STRIDE
    default_cases = [
        "baseline",
        "gate",
        "norm",
        "both",
        "atomic",
        "scaled",
        "baseline_repeat",
        "wide_grid",
        "norm_compile",
        "scaled14",
        "atomic_wide",
        "split_dw",
    ]
    cases = os.environ.get("HIP_BENCH_CASES", ",".join(default_cases)).split(",")
    for case in filter(None, cases):
        if case == "wide_grid":
            from fairchem.core.frozen_hip.nitro_candidates import (
                enable_wide_radial_grid,
            )

            enable_wide_radial_grid()
        try:
            row = bench("baseline" if case == "baseline_repeat" else case)
            report["cases"][case] = row
            print("RESULT", case, json.dumps(row), flush=True)
        except Exception:
            report["cases"][case] = {"error": traceback.format_exc()}
            print("CASE_FAILED", case, traceback.format_exc(), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        save()
        edge_kernels.half_mm_float_output = original_half_mm
        linear_kernels.quantize = original_quantize
        edge_kernels.GRID_E_STRIDE = original_grid
        if case == "wide_grid":
            edge_kernels.mixed_radial, edge_kernels.mixed_radial_grad = original_radial
    edge_kernels.mixed_radial, edge_kernels.mixed_radial_grad = original_radial
    try:
        # This test measures the benefit ceiling for one fixed batch shape. It excludes
        # input copies and does not imply that arbitrary ragged batches can replay it.
        h = configure("baseline")
        compile_head(h)
        opt = torch.optim.AdamW(
            h.parameters(), lr=0.001, weight_decay=0, fused=True, capturable=True
        )
        params = list(h.parameters())
        graph_index = int(os.environ.get("HIP_BENCH_GRAPH_INDEX", "-1"))
        d = batches[graph_index]

        def graph_update():
            opt.zero_grad(set_to_none=True)
            v = loss(h, d)
            (v * 1024).backward()
            torch._foreach_div_([p.grad for p in params if p.grad is not None], 1024)
            torch.nn.utils.clip_grad_norm_(
                params, 10, error_if_nonfinite=False, foreach=True
            )
            opt.step()
            return v

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):
                graph_update()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        begin = time.monotonic()
        for _ in range(24):
            graph_update()
        torch.cuda.synchronize()
        eager = (time.monotonic() - begin) / 24
        before = copy.deepcopy(h.state_dict())
        opt_before = copy.deepcopy(opt.state_dict())
        # Capture records kernels. Compare three replays against three eager updates.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_loss = graph_update()
        replay_rng = torch.cuda.get_rng_state()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        captured = {k: v.detach().clone() for k, v in h.state_dict().items()}
        # Keep captured storage intact. Compare to an independent eager model.
        ref = configure("baseline")
        compile_head(ref)
        ref.load_state_dict(before)
        refopt = torch.optim.AdamW(
            ref.parameters(), lr=0.001, weight_decay=0, fused=True, capturable=True
        )
        refopt.load_state_dict(opt_before)
        torch.cuda.set_rng_state(replay_rng)
        for _ in range(3):
            refopt.zero_grad(set_to_none=True)
            v = loss(ref, d)
            (v * 1024).backward()
            torch._foreach_div_(
                [p.grad for p in ref.parameters() if p.grad is not None], 1024
            )
            torch.nn.utils.clip_grad_norm_(
                ref.parameters(), 10, error_if_nonfinite=False, foreach=True
            )
            refopt.step()
        parameter_error_global = (
            (
                sum(
                    (captured[k] - v).float().square().sum()
                    for k, v in ref.state_dict().items()
                    if v.is_floating_point()
                )
                / sum(
                    v.float().square().sum()
                    for v in ref.state_dict().values()
                    if v.is_floating_point()
                )
            )
            .sqrt()
            .item()
        )
        parameter_error = max(
            rel(captured[k], v)
            for k, v in ref.state_dict().items()
            if v.is_floating_point()
        )
        del ref, refopt, captured, before, opt_before
        gc.collect()
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        replay = (time.monotonic() - start) / 100
        report["cuda_graph_fixed_shape"] = {
            "eager_seconds": eager,
            "replay_seconds": replay,
            "speedup": eager / replay,
            "parameter_relative_l2_max_after_3_updates": parameter_error,
            "parameter_relative_l2_global_after_3_updates": parameter_error_global,
            "finite_loss": bool(torch.isfinite(captured_loss)),
            "scope": "one fixed selected batch; no input copies or padding",
            "batch_index": graph_index,
        }
        print("GRAPH", json.dumps(report["cuda_graph_fixed_shape"]), flush=True)
    except Exception:
        report["cuda_graph_fixed_shape"] = {"error": traceback.format_exc()}
        print("GRAPH_FAILED", traceback.format_exc(), flush=True)
    report["frozen_verified"] = parameter_hash(model.backbone) == frozen
    save()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()

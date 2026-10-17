# Frozen UMA-s-1.2 with an additional HIP head

This workflow freezes every pretrained UMA parameter. It trains the existing
equivariant `Hessian_Head` on UMA autograd Hessians. It retains the full
intramolecular Hessian graph, including long-range pairs. It does not compute
energy, force, or autograd Hessian predictions during training.

The data source is the verified 10-million-sample
[HIP-UMA-OMol25 snapshot](https://huggingface.co/datasets/andreasburger/HIP-UMA-OMol25).
Use the snapshot manifest to download and verify each shard. Place a
`shard_XXXXX.verified.json` marker next to each verified HDF5 file. Each marker
contains its canonical `path`, `size`, `sha256`, and `samples` from the manifest.

The streaming downloader writes these markers after checksum verification:

```bash
python tools/download_verified_snapshot.py --repo "$DATASET_REPO" \
  --manifest "$SNAPSHOT_MANIFEST" --output "$DATA_DIR" --workers 32
```

Run through your scheduler on a GPU compute node:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m fairchem.core.frozen_hip.train \
  --data "$DATA_DIR" --output "$RUN_DIR" \
  --workers 2 --microbatch-pairs 196608 --max-batch 64 --max-atoms 2560 \
  --window-shards 8 --tf32 --checkpoint-edges 200000 --compile-head \
  --feature-cache "$FEATURE_CACHE_DIR"
```

Use `--nproc_per_node=1` for a single-GPU run. The optimized example below uses one GPU. The global batch limits stay the
same; gradient accumulation adjusts the number of smaller GPU operations.

All paths and scheduler settings come from the local environment. No scheduler
account, host, filesystem, or partition is required in tracked configuration.

The loader decompresses each shard once per pass. `BalancedBatchSampler` packs
mixed molecule sizes from eight shards with largest-first best-fit packing.
The example uses two GPUs for one run. `--max-batch 64` is the global molecule
cap, with at most 32 molecules per GPU. `--max-atoms 2560` targets a constant
atom count, with at most 1,280 atoms per GPU. The molecule count decreases for
large molecules. Small molecules can reach the molecule cap before the atom
budget. Window tails can contain fewer atoms. Samples are never dropped.

`--microbatch-pairs` limits each GPU forward/backward operation by the sum of
squared atom counts. The trainer accumulates gradients across these smaller
batches before one optimizer update. It preserves the global mean molecular
loss, even when ranks have different molecule and microbatch counts. Only the
last microbatch synchronizes gradients. An optional `--pair-budget` can also
limit the full optimizer batch; its default leaves the atom budget in control.

A molecule larger than a work budget forms a singleton batch. The largest
supported molecule must therefore fit one GPU. Measure peak memory before
increasing the microbatch limit. Global molecule and atom limits must divide
the GPU count exactly. Different ranks read disjoint shard groups. Training excludes a
deterministic one percent of parent indices. Validation uses those held-out
indices.
The loss gives each molecular Hessian equal weight. DDP corrects for unequal
sample counts between ranks.

The loader matches the teacher's coordinate centering before FP32 conversion.
Every rank partitions the same broadcast shard inventory. A coordinated refresh
adds new verified downloads during training. This avoids overlapping partitions
from independent directory scans while files arrive.

The optional feature cache stores frozen node features as FP32 during the
first pass. FP16 backbone inference changes their values slightly. Subsequent passes skip the backbone when a complete cache exists.
Cache namespaces include the frozen parameter hash, geometry version, backbone
precision, grouped-kernel setting, and TF32 setting.
Each cache file records its source shard hash. Incomplete caches stay in bounded
host memory and never become visible to readers. A bounded background writer
copies features through pinned host memory and publishes complete files.
`--cache-memory-gb` controls active arrays; its default is 16 GiB.
Queued host copies and loader workers need additional memory. Validation recomputes its
features. Feature caching needs additional disk space and adds first-pass I/O.

Use `--head-precision bf16-mlp` to test BF16 in the scalar radial MLPs and
scalar gating MLPs. They return FP32 coefficients to the equivariant layers.
Parameters, optimizer state, geometry, rotations, equivariant projections,
Hessian assembly, and losses retain FP32. Standard layer normalization uses
FP32 under CUDA autocast. This mode preserves checkpoint parameter names.
Use `--head-precision bf16` to test autocast across the complete head.
The default backbone mode uses FP32 tensors. The recommended mode below uses
FP16 only for frozen inference. The head keeps FP32 tensors. `--tf32` permits TensorFloat-32 matrix operations. Compare
numerical errors and throughput with `tools/benchmark_frozen_hip.py` before
selecting these options. Large graphs use activation checkpointing according
to `--checkpoint-edges`.

`--compile-head` compiles the HIP message blocks and edge readout with dynamic
shapes. It preserves checkpoint parameter names. The trainer disables DDP
compiler graph splitting because it fails on symbolic molecular graph reshapes.
Measure the initial compilation cost and steady throughput on representative
molecule sizes before enabling it.

Checkpoints contain the HIP head, optimizer state, step, frozen parameter hash,
and public model and data identifiers. The next invocation resumes `latest.pt`
automatically. It starts a new shuffled data pass. Checkpoints verify that the
backbone has no gradients and that every frozen parameter remains unchanged.
Scheduler time limits and termination signals trigger a final checkpoint.
An optional `--stop-file` requests a coordinated checkpoint and clean exit.

Training logs report throughput, loss, and peak GPU memory. Validate convergence
on the held-out split before interpreting training throughput as model quality.

`tools/benchmark_hip_precision.py` compares strict FP32, TF32, selective BF16,
and full-head BF16 on identical samples and frozen features. It measures output,
loss, and gradient errors before changing any head parameters. It also reports
end-to-end and cached-feature throughput and peak memory. Small numerical errors
alone do not prove equal validation accuracy after a complete training run.

## Fixed evaluation policy

Build the evaluation manifest once after the full snapshot download completes:

```bash
python tools/build_hip_validation.py --data "$DATA_DIR"
```

This writes `validation_10000.json` beside the verified shards. A fixed
SplitMix64 rule selects one tenth of the original one-percent parent holdout.
The ten-million-sample dataset therefore supplies approximately 10,000 molecules.
The rule does not use the training seed, GPU count, or batch settings.
Training continues to exclude the complete original one-percent holdout.

Validation runs every 100,000 optimizer steps. Every evaluation covers the
complete fixed cohort. The loader reads only selected atom and Hessian spans.
Each metric record includes the actual evaluated sample count and a SHA256
fingerprint of the sample identities. Every sample must occur exactly once.
The same manifest must be used across runs. Resume rejects a changed fingerprint.

At the training step target, the trainer saves `latest.pt`, synchronizes ranks,
and reloads that last checkpoint for final evaluation. The final record includes
the checkpoint SHA256 and the sample fingerprint. It writes
`final_evaluation.json` and appends to `evaluation.jsonl`. The terminal step
gets one final evaluation instead of duplicate periodic and final evaluations.
Allocation timeouts only save a checkpoint; they do not run final evaluation.

For an explicitly requested evaluation before the step target, use:

```bash
python -m fairchem.core.frozen_hip.train \
  --data "$DATA_DIR" --output "$RUN_DIR" --eval-only
```

This loads the output directory's `latest.pt` and evaluates the same fixed
cohort. It preserves the saved checkpoint and its training step.

### Compact frozen-feature bundles

`tools/prepare_frozen_hip.py` reads verified source shards once. It writes aligned
`.hip` files that the loader accesses through memory maps. Each bundle records
its source checksums, backbone checksum, geometry version, and precision settings.
Atomic publication and deterministic source groups allow independent workers and
safe restarts. Keep source shards for evaluation and scientific provenance.

```bash
python tools/prepare_frozen_hip.py --data "$DATA_ROOT" --output "$PACKED_ROOT"
python -m fairchem.core.frozen_hip.train --data "$DATA_ROOT" \
  --packed-data "$PACKED_ROOT" --output "$RUN_ROOT" --tf32 --compile-head \
  --fused-rotations --block-so2 --projected-readout
```

The default feature storage is FP32. Optional `--feature-dtype float16` uses an
exact power-of-two scale before quantization. The loader restores FP32 features
on the GPU. This option changes the features slightly; measure its errors before
using it for a scientific run. Hessian targets remain FP32.

The label format stores off-diagonal atom-pair blocks once and six entries per
node diagonal block. It symmetrizes labels and retains the antisymmetric squared
error as a constant. Weighted block loss therefore preserves the original dense,
equal-molecule MSE and its parameter gradients, even for asymmetric source data.
The format uses offsets, with no molecule-size padding.

`--fused-rotations` changes the HIP head's parameterless operation dispatch. It
retains general composition-dependent mixture routing and all model parameters.
It does not enable the merged-expert inference backend. Geometry derivatives are
unnecessary for frozen-coordinate head training; rotation kernels skip their
saved inputs and backward work when those derivatives are not requested.

`tools/benchmark_packed_hip.py` compares identical molecules, starting weights,
and update operations. It reports preprocessing separately from training speed,
checks parameter gradients, and records GPU telemetry. `tools/profile_frozen_hip.py`
separates data loading, frozen inference, and cached-head updates.


## Tested fast configuration

Prepare compact labels on CPU. Independent workers can share the output directory:

```bash
python tools/prepare_frozen_hip.py --data "$DATA_ROOT" --output "$PACKED_ROOT" \
  --targets-only --rank "$WORKER_INDEX" --world "$WORKER_COUNT"
```

Use the same worker count and disjoint worker indices for one conversion pass.
The completed training split contains 9,900,330 molecules in 5,239 bundles.
Compact labels and geometry occupy about 730 GB. The original snapshot occupies
about 1,335 GB. Keep the original files for fixed evaluation and provenance.
Feature caches require separate storage.

The following settings passed training benchmarks on one 80 GB GPU:

```bash
python -m fairchem.core.frozen_hip.train \
  --data "$DATA_ROOT" --packed-data "$PACKED_ROOT" --output "$RUN_ROOT" \
  --workers 2 --max-batch 64 --max-atoms 2560 --microbatch-pairs 524288 \
  --tf32 --backbone-precision fp16 --head-precision fp32 \
  --fused-rotations --block-so2 --projected-readout --compile-head \
  --feature-cache "$FEATURE_CACHE_ROOT" --cache-memory-gb 16 \
  --validate-every 100000 --steps 200000
```

The microbatch limit of 524,288 is specific to the tested 80 GB memory budget.
The portable default remains 196,608. Four 350-atom molecules used 49.5 GB of
allocated memory and 50.6 GB of reserved memory in a stress test.
The normal training benchmark used 34.8 GB allocated and 36.2 GB reserved.
The tested balanced batches required one microbatch per optimizer update.
Larger batches can still require accumulation.

The default activation-checkpoint threshold now exceeds the microbatch pair
limit. This avoids recomputing batches that already fit the selected memory
budget. Use an explicit `--checkpoint-edges` value if recomputation is necessary.

The fused kernels construct ragged graphs and compute the compact block loss.
The rotation backward kernel scatters node gradients directly. It avoids a large
edge-gradient intermediate. Atomic addition changes floating-point summation
order. Deterministic mode uses the reference gradient path.

`--block-so2` forms complex block weights through autograd on every update.
`--projected-readout` contracts the final convolution with the scalar projection
before rotation. Both retain the original trainable parameters and optimizer
state. Neither changes the head architecture.

`--grouped-mole` provides an optional frozen FP32 inference kernel. It preserves
composition-dependent expert routing. Ordinary FP16 inference was faster in the
tested workload, so the recommended command omits this option.

### Measured results

The controlled comparison uses identical starting weights and 442 molecules.
They contain 38,496 atoms across 24 physical batches. Timings include head
backward, clipping, and optimizer updates. Compilation and preprocessing are
measured separately.

| Mode | Molecules/second |
| --- | ---: |
| Original compiled head, fresh FP32 backbone features | 25.1 |
| Optimized head, fresh FP16 backbone features | 91.5 |
| Original compiled head, existing frozen features | 116.8 |
| Optimized head, existing frozen features | 199.1 |

Fresh-feature throughput improves 3.65 times. Cached-head throughput improves
1.70 times. Cached results exclude feature preparation; they must not be compared
with fresh-feature results as a complete training speedup.

A separate full-trainer benchmark uses compact files and asynchronous cache
writes. It averages 179 molecules/second and 0.245 seconds per optimizer step
after warmup. GPU activity averages 78%, with an 86% median. This workload has
smaller molecules than the controlled comparison. It has no feature-cache hits.
Compilation, queue time, and future cache hits change total runtime.
At this measured step time, 200,000 updates take about 13.6 hours, excluding
startup and evaluation. This estimate does not establish time to convergence.

The precision check evaluates the same 10,027 held-out molecules and checkpoint.
FP32 inference with TF32 enabled gives MSE 1.3655248 and MAE 0.23301139. Frozen FP16 inference with
the optimized FP32 head gives MSE 1.3653877 and MAE 0.23302135. MAE changes by
0.0043%; prediction relative L2 error is about 0.0714%. These results check
checkpoint evaluation. They do not establish identical convergence after training.

Full-head FP16 produced nonfinite gradients in testing. Keep the head and loss
in FP32. Optional FP16 feature storage uses power-of-two scaling, but the
recommended feature cache stores FP32. Target Hessians always retain FP32.

Use `--benchmark` with a separate output directory for bounded speed tests.
Benchmark mode does not save checkpoints or run periodic validation. Production
training remains a separate, explicit action.

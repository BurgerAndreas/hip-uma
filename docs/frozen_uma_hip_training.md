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

Cache the fixed cohort once to avoid thousands of small source-file reads:

```bash
python tools/cache_hip_validation.py --data "$DATA_DIR" --workers 8
```

The loader automatically reads `validation_10000.hip` when this file exists.
The cache keeps original dense labels and coordinates. It does not symmetrize
validation targets, so MAE and MSE retain their original definitions.
Its metadata binds source checksums, selected rows, and the fixed sample manifest.
A different manifest causes an error. A missing cache uses the source files.
The cache retains source-entry partitions across GPUs and loader workers.
The full cohort cache occupies 1.46 GB. A checkpoint evaluation takes 64.4
seconds with the cache, versus 333.7 seconds from source shards. Both runs
evaluate all 10,027 samples. The two runs differ by 0.0017% in MAE. CPU tests confirm exact preservation
of source coordinates and labels.

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
  --loss block-rmse --tf32 --backbone-precision fp16 --head-precision fp32 \
  --backbone-graph-version 4 --grouped-mole --fused-backbone \
  --fused-rotations --block-so2 --projected-readout --compile-head \
  --feature-cache "$FEATURE_CACHE_ROOT" --feature-cache-read-only \
  --validate-every 100000 --steps 200000
```

The microbatch limit of 524,288 is specific to the tested 80 GB memory budget.
The portable default remains 196,608. Four 350-atom molecules used 49.5 GB of
allocated memory and 50.6 GB of reserved memory in a stress test.
An earlier training benchmark used 34.8 GB allocated and 36.2 GB reserved.
The current matched block-RMSE benchmark peaks at 47.1 GB allocated and
48.8 GB reserved. Most updates use one microbatch; some require accumulation.
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

`--grouped-mole` preserves composition-dependent expert routing. FP16 inference
packs merged expert weights into contiguous storage once per layer. This avoids
repeated strided FP32 loads across row tiles. `--fused-backbone` also uses fused
rotation kernels for frozen inference. The head has separate operation dispatch.
All pretrained parameters remain frozen.

### Measured results

The controlled comparison uses identical starting weights and 442 molecules.
They contain 38,496 atoms across 24 physical batches. Timings include head
backward, clipping, and optimizer updates. Compilation and preprocessing are
measured separately. These controlled timings exclude source-file loading.

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


## Block RMSE objective

Use `--loss block-rmse` for equal-geometry averaging of Cartesian block norms.
For each geometry, the loss sums diagonal Frobenius norms and twice the
independent off-diagonal Frobenius norms. It divides this sum by three times
the squared atom count. It then averages across geometries.

The inputs from the HIP head are raw directed blocks. Assembly adds each
reverse transpose before calculating the error. Diagonal reconstruction includes
all nine Cartesian components, even though storage keeps six independent entries.
The fused CUDA path uses no molecule loop, dense Hessian assembly, or size padding.

The objective uses symmetrized targets from the compact training files.
The antisymmetric MSE constant does not apply to block RMSE. It is ignored for
this objective. Source validation still reports original Cartesian MAE and MSE.
It also reports block RMSE against symmetrized labels.
All 10,027 original validation Hessians are exactly symmetric in the checked cohort.

Norm reductions and gradients use FP32. The kernel rescales each error block
before squaring to reduce overflow and underflow. It adds no epsilon.
Exact agreement produces a zero loss and zero gradients.
Tests cover unequal molecule sizes, diagonal weights, off-diagonal weights,
rotation invariance, extreme scales, and agreement with dense block gradients.

The default loss remains MSE for compatibility with existing runs.
Use a separate output directory when changing the objective. The trainer rejects
a checkpoint with a different loss during normal training. Benchmark mode permits
that comparison without saving updates. Checkpoints record the objective and
inference settings. Evaluation restores those settings before loading the model.

## GPU idle time and profiling

The original trace records 4,379 scalar extractions inside graph generation
across eight optimizer updates. The grid graph builder loops over geometries and
reads GPU scalars while building periodic-cell offsets. These molecules have no
periodic boundaries. Most of that setup is unnecessary for this dataset.

`--backbone-graph-version 1` selects the vectorized all-pairs radius builder.
The cutoff and neighbor limit remain unchanged. Graph tests compare edge sets
and displacement vectors, including neighbor pruning and 350-atom molecules.
The all-pairs method uses quadratic work. The microbatch pair limit bounds this
work for the supported dataset. The default graph version remains two.

Pinned-memory loading now uses the custom `AtomicData.pin_memory` method.
Without this method, PyTorch leaves this custom class unpinned.
`--no-pin-memory` supports controlled comparisons. The trainer keeps periodicity
flags on the CPU because graph generation uses them for Python branches.
Single-GPU stop checks also stay on the CPU. Metric counts transfer only when
logging. Multi-GPU sample weighting and coordinated stopping remain unchanged.

Feature-cache namespaces include the graph version and frozen rotation backend.
Embedded feature bundles reject mismatched settings. Existing target-only
bundles remain compatible and require no conversion.

Capture a bounded trace after compilation and warmup:

```bash
python -m fairchem.core.frozen_hip.train \
  --data "$DATA_ROOT" --packed-data "$PACKED_ROOT" --output "$BENCHMARK_ROOT" \
  --benchmark --loss block-rmse --duration 360 --profile-after 100 --profile-steps 8 \
  --tf32 --backbone-precision fp16 --backbone-graph-version 1 \
  --grouped-mole --fused-backbone --fused-rotations --block-so2 \
  --projected-readout --compile-head
python tools/summarize_hip_profile.py "$BENCHMARK_ROOT/profile-rank0.json"
```

The summary reports kernel activity, CPU phases, and scalar extractions.
Phase times include nested operations. Profiling adds CPU overhead, so use
unprofiled update times and GPU telemetry for final speed comparisons.
GPU activity measures time with active kernels; it does not measure peak
arithmetic efficiency. Small batches can still leave hardware unused.
Keep raw traces private because they include runtime file paths.

### Matched block-RMSE benchmark

These results use one A100 80 GB GPU and the same optimizer updates,
from step 12,200 through 12,720. Both runs use block RMSE and FP16 frozen
inference. Their molecule counts and atom counts match at each logged window.
The comparison excludes compilation and initial warmup. No frozen features
come from cache in these windows.

| Setup | Molecules/s | Seconds/update | Mean GPU activity | Median GPU activity |
| --- | ---: | ---: | ---: | ---: |
| Previous graph and expert dispatch | 186.9 | 0.2610 | 70.0% | 76% |
| Vectorized graph, packed experts, fused body, pinned loader | 345.4 | 0.1413 | 88.4% | 95% |

This is a 1.85 times throughput increase on the matched sequence.
The bounded trace reduces kernel count from about 40,000 to 12,782 across
eight updates. Graph scalar extractions fall from 4,379 to 234.
The loader takes 0.48 ms across those updates in the optimized trace.
A separate GPU-side finite-gradient check gives 343.2 molecules/s on the same
sequence. It does not improve throughput, so the trainer retains its normal check.

The combined frozen FP16 path also passes the full 10,027-sample evaluation.
Against the original FP32 path, MAE changes from 0.23301611 to 0.23302137
(0.0023%). Prediction relative L2 error is 0.0622%.
Both evaluations use the same step-12,000 checkpoint and fixed sample identities.
This checks inference accuracy; it does not establish equal training convergence.

Small kernel launches and remaining graph scalar reads still create gaps.
Balanced batches also vary in pair count and computation, even at similar
atom counts. GPU activity therefore varies between batches. Activity above
90% in many samples does not establish 90% arithmetic efficiency.

At 0.1413 seconds per update, 200,000 updates need about 7.8 hours.
This estimate excludes compilation, queue time, validation, and checkpoint I/O.
Different molecular-size sequences and feature-cache hits change this estimate.

### Fused molecular radius graphs and expert tiles

`--backbone-graph-version 4` selects a frozen molecular graph kernel.
It supports nonperiodic FP32 coordinates and up to 1,024 atoms per molecule.
One pass selects neighbors. A second pass writes compact edges and distances.
Small padded masks serve as temporary scratch space. Hessian storage and
returned graph edges still use indices without size padding.

The kernel preserves the reference cutoff, near-zero distance exclusion,
and non-strict neighbor rule. That rule uses sorted squared distance at the
zero-based neighbor limit, then adds 0.01. Strict neighbor pruning uses the
reference implementation. Graph partitioning and coordinate gradients remain
unsupported by the fused path. Tests include coincident atoms, cutoff ties,
neighbor pruning, single atoms, and empty graphs.

The bounded graph trace reduces graph CPU time from 41.5 ms to 5.5 ms
across eight updates. Device-to-host copies across the trace fall from 260
to 44. Full training comparisons must also check resulting throughput.

FP16 grouped expert contractions now use 64 by 128 output tiles and
a reduction width of 64. FP32 retains its previous tiles.
The tile metadata includes the row width, so precision switches cannot reuse
incompatible prefix counts. On six captured expert shapes, aggregate kernel
time falls from 3.81 ms to 1.72 ms. Their FP16 outputs match the previous tiles
exactly in this check. Full training benchmarks measure the resulting throughput
separately.

Some dimers exceed FP16 range in the second expert convolution.
The grouped backend keeps its matrix inputs in FP16 and retains FP32 outputs
through inverse rotation. This avoids an overflow without a per-batch CPU check.
The cache namespace records this output precision. Embedded feature bundles
must also record `expert_fp32_output`. Cache publication rejects nonfinite values.
Legacy cache reads check finiteness when their metadata lacks this guarantee.

The safe mixed-precision configuration reaches 376.3 molecules/s in the
full-stream benchmark, with 90.4% mean GPU activity and a 96% median.
Peak allocated memory remains 47.1 GB. The measurement uses the same
step-12,200 through step-12,720 molecular sequence as the earlier benchmark.
Shared-cluster timings vary. Controlled repeat timings also show this variation.

GPU kernel correlation assigns 68.6% of execution time to the HIP head and
its backward pass. Frozen inference accounts for 26.9%; loss kernels account
for 0.03%. These percentages exclude memory copies and idle time.
The head's matrix operations and memory traffic offer the largest remaining
kernel targets.

### Prepare frozen features without reading Hessian targets

`tools/cache_packed_hip_features.py` reads existing compact geometry directly.
It skips Hessian arrays on the CPU and leaves targets off the GPU.
It publishes the same FP32 feature cache that training already reads.
Original HDF5 files and compact target files need no conversion.

Run independent preparation workers with disjoint rank numbers:

```bash
python tools/cache_packed_hip_features.py \
  --packed-data "$PACKED_ROOT" --feature-cache "$FEATURE_CACHE_ROOT" \
  --tf32 --backbone-precision fp16 --backbone-graph-version 4 \
  --grouped-mole --fused-backbone --rank "$PREP_RANK" --world "$PREP_WORLD"
```

Use the same world size and distinct ranks for one preparation pass.
Each process uses one GPU. Completed bundles survive restarts.
The tool verifies each assigned bundle and reports missing caches.
If memory limits prevent complete publication, reduce the window size or
increase cache memory. Rerun preparation to fill missing bundles.
`--max-files` and `--duration` support bounded checks.

Training must use the same inference flags and feature-cache root.
`--feature-cache-read-only` consumes complete caches without creating new ones.
It still computes frozen inference for missing features.
The default continues to publish features during training.

The file-backed comparison uses 53,880 small molecules in 17 bundles.
On the same GPU and matched updates, online inference reaches 1,084 molecules/s.
Prepared features reach 3,792 molecules/s, a 3.50 times update speed increase.
These small-molecule rates depend on the cohort. Feature preparation remains
a separate cost. Caching helps when experiments reuse features or preparation
runs in parallel. The complete FP32 feature cache needs about 2.04 TB.

The safe path passes evaluation on all 10,027 fixed samples and the previously
failing dimer batch. Cache checks verify 53,880 finite samples and successful
reuse without recomputation. Production training remains a separate action.

## Additional kernel and input optimizations

`--segmented-scatter` combines inverse Wigner rotation and node accumulation.
It reduces target-sorted messages without atomics or a wide rotated intermediate.
Its backward kernel combines the corresponding gradient gather and rotation.
Use it with `--fused-rotations` and a complete intramolecular HIP graph.
`--fused-scatter` provides an atomic alternative for other edge orderings.
Geometry derivatives use the original implementation.

The rotation gather uses 64-channel tiles. Inverse rotation keeps 128-channel tiles.
Launch grids adapt to small edge counts. Captured rotation outputs remain bitwise identical.
Gather kernel measurements improve by 10–15%. These figures describe individual kernels, not full training.

`--async-grad-check` checks the gradient norm on the GPU before the optimizer update.
It removes a CPU synchronization from each update. Nonfinite gradients still stop training.
`--prefetch-factor 8` buffers more host batches per loader worker.
Dense HIP heads also skip unused expert-routing transfers.
The balanced sampler retains its original ordering and exact sample coverage.
A 20,000-molecule packing test reduces CPU time from 0.95 seconds to 0.017 seconds.
This packing improvement does not represent a full-training speedup.

One matched full-stream test covers 3,600 optimizer updates after warmup.
All modes use the same molecules, atom counts, and uncached frozen inference.

| Mode | Molecules/s | Seconds/update | Mean GPU activity |
| --- | ---: | ---: | ---: |
| Previous configuration | 302.4 | 0.14624 | 89.1% |
| GPU gradient check | 319.2 | 0.13854 | 94.3% |
| Gradient check and fused atomic accumulation | 323.1 | 0.13683 | 93.9% |
| Gradient check and fused segmented accumulation | 327.4 | 0.13504 | 94.0% |

The final row improves throughput by 8.3% on this sequence.
Feature preparation can remove frozen inference from subsequent training updates.
Shared hardware and molecular sizes affect measured times.
See `frozen_hip_kernel_benchmark.json` for controlled measurements and limitations.

`--scaled-linear` tests power-of-two scaled FP16 matrix operands with FP32 outputs.
Parameters, gradients, optimizer state, and reductions remain FP32.
The implementation protects long weight-gradient reductions from FP16 overflow.
A compiler wrapper supports the FP32 output API on older compiler versions.
This mode saves about 5.2 GB in the controlled head benchmark.
It reduces throughput by about 20%, so the recommended speed configuration disables it.

`--fused-wigner` tests a compact quartic polynomial kernel for frozen geometry.
It reduces the polynomial basis from 256 redundant products to 35 monomials.
It also calculates the inverse rotation envelope.
The full fixed cohort changes MAE by approximately 0.005% at the tested checkpoint.
The standalone head benchmark shows no speed gain. Keep this option experimental.

## MAE comparison and long experiments

`--loss mae` uses equal-geometry averaging of Cartesian absolute errors.
Packed labels remain symmetrized. Independent off-diagonal blocks receive weight two.
Diagonal blocks include all nine components. Exact agreement gives a zero gradient.
The fused kernel avoids dense Hessian reconstruction and retains FP32 arithmetic.
MAE and block RMSE share the same molecular weights and packed label format.

Use separate output directories for the two losses. Both experiments can start from the same seed.
Checkpoints record the initial head checksum and the target step count.
Set `CODE_REVISION` to the committed Git revision. Checkpoints also record this revision.
Resume rejects a different target step count during production training.

`scripts/train_frozen_hip_segment.sh` supports experiments longer than one scheduler allocation.
It reads private paths and scheduler settings from environment variables.
Each allocation resumes `latest.pt` and retains the original optimizer schedule.
One dependent successor waits for successful completion of the current allocation.
The script queues that successor before training starts. It never polls scheduler status.
A controller outage therefore does not remove an accepted successor.
If submission fails, the script tries once after saving the completed segment.

Set these variables in a private scheduler wrapper:

```bash
export PYTHON_BIN CODE_ROOT DATA_ROOT PACKED_ROOT RUN_ROOT LOSS_NAME
export FEATURE_CACHE_ROOT CONTINUATION_SCRIPT
export TARGET_STEPS=2000000
export SEGMENT_SECONDS=13200
export ASYNC_GRAD_CHECK=1
export SEGMENTED_SCATTER=1
bash "$CODE_ROOT/scripts/train_frozen_hip_segment.sh"
```

`CONTINUATION_SCRIPT` points to that private scheduler wrapper.
The wrapper must request one GPU and provide the required environment.
It must include its account, partition, time, and memory settings for successor submissions.
The default batch cap is 64 molecules, with an atom target of 2,560.
`MICROBATCH_PAIRS` changes only the physical GPU work limit.
It preserves the global batch and equal-geometry loss weights.
`MAX_SEGMENTS` defaults to 128 allocations as a continuation limit.

An allocation lock prevents concurrent updates to one experiment.
A `STOP` file prevents further training and cancels its queued successor.
Termination signals request a checkpoint. Scheduler requeues replace dependencies that an earlier failure can invalidate.
Repeated starts within one allocation reuse its accepted successor.
An explicit `STOP` file ends continuation. Failed training does not advance its dependency chain.
After two million updates, training reloads its last checkpoint for final evaluation.
Validation retains the fixed 10,027-sample cohort and 100,000-update interval.
Completed experiments cancel their unused successor and reject duplicate execution.

The allocation script saves every 5,000 updates. `CHECKPOINT_EVERY` can change this interval.
Hashing the 1.17 GB frozen backbone takes approximately one second in the measured environment.
Saving the 35 MB head and optimizer takes approximately 0.05 seconds.
The larger interval reduces checkpoint overhead. Signals and allocation endings still request an immediate checkpoint.

A preparation inventory can restrict workers to unfinished bundles:

```bash
python tools/cache_packed_hip_features.py \
  --packed-data "$PACKED_ROOT" --feature-cache "$FEATURE_CACHE_ROOT" \
  --inventory "$INVENTORY_JSON" --workers 0 --window-files 1 \
  --tf32 --backbone-precision fp16 --backbone-graph-version 4 \
  --grouped-mole --fused-backbone --rank "$PREP_RANK" --world "$PREP_WORLD"
```

The inventory contains a fixed JSON list of relative `.hip` filenames.
Every worker must use the same inventory and worker count.
This permits disjoint repair passes without repeating completed preparation.
Zero loader workers avoid multiprocessing tensor transfers during repair.
A one-file window limits incomplete feature arrays.
Progress records follow processed-sample counts. They do not query scheduler status.

### Fused radial messages and the complete feature cache

`--fused-radial` combines node gathering, Wigner rotation, and radial weighting.
Its backward kernel recomputes rotations from node features.
It avoids storing the complete unweighted edge tensor.
Geometry, message arithmetic, gradients, and reductions remain FP32.
The full fixed cohort changes MAE by less than 0.001% at the tested checkpoint.

A controlled 3,327-molecule comparison gives these results:

| Mode | Molecules/s | Peak allocated GB |
| --- | ---: | ---: |
| Reference compiled head | 841.8 | 45.65 |
| Fused radial messages | 843.4 | 40.76 |
| Fused radial messages and segmented accumulation | 887.7 | 40.75 |
| Reference repeat | 841.0 | 45.65 |

The radial kernel saves approximately 4.9 GB while retaining throughput.
The combined configuration improves controlled head throughput by approximately 5.5%.
Numerical comparisons do not establish equal training convergence.

The prepared cache covers all 9,900,330 training samples across 5,239 bundles.
It stores 2.04 TB of FP32 features for 443,614,295 atoms.
Verification checks each source identity, array shape, and publication record.
The complete training split contains between two and 350 atoms per geometry.
Training reads complete caches without running the frozen backbone again.
Validation continues to recompute its own features.

On the tested 80 GB GPU, the complete atom budget fits one physical update.
Seven 350-atom geometries and one 110-atom fragment use 2,560 atoms.
Their squared atom counts sum to 869,600.
The stress test uses AdamW and the fused FP32 head.
It peaks at 69.25 GB allocated and 77.22 GB reserved.
The fragment serves only as a memory stress input.
A work limit of 900,000 therefore covers the verified production atom budget.
It retains the global cap of 64 geometries and the balanced atom sampler.
It requires no gradient accumulation for this verified dataset and configuration.

The production work limit remains 524,288. This setting gives higher measured throughput and leaves more memory free.
Large logical batches use gradient accumulation. Their global loss weights remain unchanged.

For the allocation wrapper, use:

```bash
export MICROBATCH_PAIRS=524288
export FUSED_RADIAL=1
export SEGMENTED_SCATTER=1
export DEVICE_PREFETCH=1
export ASYNC_GRAD_CHECK=1
```

Device prefetch transfers one logical batch on a separate CUDA stream.
Stream events protect tensor lifetimes. It does not change sample order or loss weights.
The buffered batch adds input storage. The head retains its physical work limit.


### Final matched training measurement

Thirteen matched windows cover steps 12,300 through 13,500, with 100 updates per window.
Every mode processes the same molecules and atoms.

| Mode | Molecules/s | Seconds/update | Peak allocated GB |
| --- | ---: | ---: | ---: |
| Frozen inference during training | 406.9 | 0.12071 | 47.07 |
| Prepared features, warm reference | 688.9 | 0.07131 | 47.08 |
| Prepared features and selected kernels | 756.8 | 0.06491 | 42.05 |
| Larger physical work limit | 728.4 | 0.06744 | 67.83 |

The selected configuration improves throughput by 9.9% against the warm cached reference.
It gives 1.86 times the throughput of the online reference.
The first cached reference includes cold effects and does not define the speedup.
The selected configuration records 99.5% mean GPU activity during its measured interval.
GPU activity does not measure the fraction of peak arithmetic throughput.

Two million updates need about 36 hours of measured update time at this rate.
Feature preparation, checkpoint costs, validation, compilation, and scheduler delays add time.
The two losses can have different update times during actual training.

## Mass-weighted, Eckart-projected validation metrics

Every validation and final evaluation records four additional metrics:

- `mass_weighted_eckart_abs_cos_v1`
- `mass_weighted_eckart_eigenvalue_mae`
- `mass_weighted_eckart_hessian_mae`
- `mass_weighted_eckart_hessian_rmse`

The evaluator symmetrizes both Hessians and applies inverse square-root atomic masses.
It removes rigid translations and rotations with an SVD of the mass-weighted geometric generators.
Linear molecules have five rigid modes. Nonlinear molecules have six.
All projection and eigenspectrum calculations use FP64.

The Hessian errors use the complete projected Cartesian matrices, with shape `3N × 3N`.
They use `P M^(-1/2) H M^(-1/2) P`, where `P` removes the rigid modes.
Hessian RMSE takes one square root over each complete matrix.
The evaluator then averages each metric equally across geometries.
Eigenvalue MAE first averages sorted vibrational eigenvalue errors within each geometry.
The first mode has the lowest algebraic eigenvalue, including negative eigenvalues.
The cosine uses its absolute value to remove eigenvector sign ambiguity.

Eigenvalue MAE, Hessian MAE, and Hessian RMSE use `eV/(Angstrom^2*amu)`.
The records identify the matrix frame, precision, averaging, and eligible sample counts.
Geometries without vibrational modes receive an explicit count and do not enter these means.
A coverage check verifies the complete fixed cohort before results are published.

The existing Cartesian metrics remain available under their original names.
Validation still runs every 100,000 updates on the same 10,027 geometries.
Final evaluation still reloads `latest.pt`.

Weight decay is configurable with `--weight-decay` (default `0`). The
allocation runner accepts `WEIGHT_DECAY=0` for a zero-decay ablation and passes
it to each successor. Use a separate output directory and the original seed
for a fresh comparison. Resume rejects optimizer states with different decay.
The run configuration and checkpoints record the selected decay.

New runs default to zero weight decay and validation every 400,000 updates.
The allocation runner accepts `VALIDATE_EVERY` to set the validation interval.
A four-million-step experiment sets `TARGET_STEPS=4000000`; the cosine
schedule spans that full target. Final evaluation still uses the last checkpoint.

Mass-weighted Block RMSE uses `--loss mass-block-rmse` or
`LOSS_NAME=mass-block-rmse`. Each atom-pair norm receives a factor
`1 / sqrt(m_i * m_j)`, with ASE masses in amu. Diagonal blocks receive
`1 / m_i`. Geometries retain equal weights and off-diagonal blocks retain
weight two. Targets remain symmetrized complete Cartesian Hessians, including
Coulomb terms. This loss does not apply Eckart projection. Validation retains
the same raw and mass-weighted, Eckart-projected metrics.

The fused FP32 loss applies mass factors directly to block weights in both
forward and backward kernels. It retains packed targets and avoids dense
mass-weighted Hessian allocation. No epsilon or geometry-dependent
renormalization changes the objective. The loss unit is eV/(Angstrom^2*amu).

Optional loader throughput switches preserve the sampled geometries and FP32
features. `--threaded-prefetch` moves host retrieval and asynchronous device
transfers to a bounded background producer. It requires `--device-prefetch`.
`--fast-packed-loader` gathers cached fragments directly into one batch instead
of constructing and concatenating intermediate AtomicData objects. Cache-writer
metadata and uncached inputs retain the original packing path. The allocation
runner exposes `THREADED_PREFETCH=1` and `FAST_PACKED_LOADER=1`. Enable these
switches only after matched throughput and correctness checks on the workload.

`--packed-window-memory-gb` optionally stages cached feature and target arrays
with sequential reads before random per-batch gathers. Its default is zero.
The byte budget applies separately to each loader worker and its current shard
window. Arrays that exceed the remaining budget stay memory mapped. This keeps
all samples and dtypes intact while bounding extra private host memory. Set
`PACKED_WINDOW_MEMORY_GB` in the allocation runner to request this option.

The fused FP32 radial kernels accept `--radial-tile` (`64`, `128`, or `256`)
without changing weights or tensor layouts. The default remains `64`.
`RADIAL_TILE` exposes the same choice through the allocation runner. Select
larger tiles only after forward and gradient checks and a complete training
benchmark; isolated kernel gains do not establish a training speed gain.

Keep the Python environment, base interpreter, generated kernels, bytecode,
and compiler caches on local storage. A free local filesystem can remain
available even when another local scratch filesystem is full. Shared-filesystem
fallbacks can stall imports and lazy CUDA module loading while the GPU sits idle.
`tools/stage_frozen_hip_runtime.py` stages an environment archive and relocates
its base interpreter under a caller-supplied directory. It locks publication,
checks archive identity, and verifies the relocated virtual environment before
marking it ready. Configure CUDA, Triton, and compiler cache directories beside
the staged runtime. Keep experiment checkpoints on persistent storage.


Allocation successors use `afterany` so a scheduler timeout does not discard
accepted continuation work. A handled trainer failure cancels its successor;
it does not repeatedly retry numerical errors. The `STOP` file still cancels
continuation explicitly. A scheduler interruption can resume only through the
trainer's normal checkpoint identity and optimizer checks.


`PREFETCH_FACTOR` controls the existing per-worker host queue in the allocation
runner (default `8`). A deeper queue can overlap the next shard window's reads
with more GPU updates. Keep the worker count fixed when comparing sample order,
and budget host memory for both staged windows and queued logical batches.

Storage and loader measurements on an 80 GiB Ampere GPU used identical batch
identities, atom counts, targets, and FP32 head parameters within each cohort.
The balanced atom target stayed at 2,560, with a 64-geometry cap and a 524,288
quadratic work limit. A second cohort reversed the test order on another node.
Two hundred startup updates were excluded. Any later 100-update window with a
GPU step span above one second was excluded from every run in that cohort,
so lazy compilation and allocation stalls did not enter the steady comparison.
Raw timing records retained those windows separately.

| Cohort and storage condition | Setup | Samples/s | Host batch wait, ms/update |
| --- | --- | ---: | ---: |
| First cohort, storage-sensitive control | Original loader | 511 | 67.24 |
| First cohort, after staging | Background loader, depth 32 | 880 | 0.31 |
| First cohort, warmed control | Original loader | 837 | 3.92 |
| Reversed cohort, fresh reads | Staging, depth 64 | 801 | 4.05 |
| Reversed cohort, warmed control | Original loader | 853 | 0.48 |

These numbers show strong cache dependence rather than a uniform speedup.
The selected storage profile stages at most 4 GiB per worker, uses two workers,
a host prefetch depth of 64, background device transfer, and single-pass cached
packing. Runtime libraries and generated kernels use local storage. Isolated
larger radial tiles improved individual kernels, but complete-step measurements
did not show a consistent additional gain; the selected radial tile stays 64.
The data profile adds at most 8 GiB of staged private host memory, plus queued
batches. Measured peak GPU allocation stayed near 41.6 GB in the reversed cohort.
The head, gradients, reductions, and optimizer states retain FP32 precision.


`--packed-window-prefetch` loads one future shard window on a background thread
inside each loader worker. `PACKED_WINDOW_PREFETCH=1` exposes this option through
the allocation runner. The option defaults to off. Window order, batch order,
source checks, targets, and arithmetic stay unchanged. A worker can retain its
current window and one future window. Budget twice the per-window staging memory
per worker, plus queued batches. With two workers and a 4 GiB window budget,
staged private memory can reach 16 GiB. Loading errors propagate to the consumer.
Compare complete training steps across multiple window boundaries before enabling
this option. Low aggregate CPU use does not rule out a single-thread launch limit.

A ten-second process sample during training used about 1.2 to 1.4 CPU cores
per run, including two loader workers. Each run allocated eight cores. The main
process used about one core. More loader workers do not address that launch
thread or GPU matrix work. A short warm GPU trace attributed about 44% of kernel
time to matrix products. Treat reduced-precision matrix kernels as an accuracy
experiment, and test complete-step speed before selecting them. Existing BF16
MLP and scaled FP16 variants did not consistently improve full training speed.


`--head-precision fp16-edge` uses FP16 hidden activations and tensor-core operands
inside the edge networks. It requires fused rotations, fused radial operations,
block SO2, projected readout, and segmented scatter. Use `HEAD_PRECISION=fp16-edge`
and `LOSS_SCALE=1024` through the allocation runner. The default remains FP32.

Radial gathering and inverse-rotation gradients write FP16 directly, avoiding
separate full-size casts. Linear bias addition uses the vendor GEMM epilogue.
Normalization statistics, node accumulation, the final thin projection,
Cartesian outputs, Coulomb terms, and losses retain FP32. Parameters, parameter
gradients, and AdamW state also retain FP32. Long weight-gradient reductions
use native FP32 GEMM outputs. Frozen inference retains its original reduction
policy when the head selects stricter FP16 accumulation settings.

The trainer scales the scalar loss before backward and divides all parameter
gradients before clipping and AdamW. The scale defaults to 1 for other modes
and 1024 for FP16 edge mode. The checkpoint and run configuration record it.
The existing nonfinite gradient guard rejects unsafe updates. This mode changes
rounding, so compare predictions and projected metrics before selecting it.
Sampler settings, target normalization, and the loss definition stay unchanged.


A matched step benchmark used eight microbatches with 86 geometries and identical
FP32 cached features, targets, initial parameters, and sample order. The batch
cap was 16, the atom target was 768, and the quadratic limit was 90,000.
Repeated FP32 controls measured 355.8 and 356.0 samples/s. Selective FP16 with
FP32 final projection measured 529.5 samples/s, a 1.49x gain. Peak allocated
memory changed from 7.41 GB to 4.57 GB. These are short matched-step measurements.
Production speed also depends on geometry sizes and data arrival.

Prediction relative L2 drift was 0.026% on the first real microbatch. The
350-atom synthetic stress case showed 0.031% drift and finite backward gradients.
Parameter-gradient drift ranged from 1.3% to 3.0% across the three loss variants.
An 86-geometry numerical check covered five trained checkpoints. Projected
error increases stayed below 1%; the largest abs-cos decrease was 0.0012.
That check used training examples. It does not replace fixed-cohort validation.

GPU activity can remain near 100% on warm batches, then fall during shard reads.
Short activity samples do not establish a sustained utilization average.
After compute kernels become faster, profile data arrival across window
boundaries and compare prefetch settings on multiple storage conditions.


A production canary with parallel window loading averaged 98.8% GPU activity
across 120 one-second samples, with no zero-activity samples. Its first 1,900
updates after startup averaged about 902 samples/s. The staging budget stayed
at 4 GiB per worker per window, with two workers. Earlier 30-second samples
without window overlap averaged 57% and 75% activity on two runs. These live
windows used different molecular batches, so they are not a matched speed ratio.
Keep the matched kernel result separate from the loader observations.

### Translation sum rule

`--translation-projection` applies `P H P` to each complete Cartesian Hessian,
where `P = (I - 11^T/N) tensor I_3`. This removes the three translation modes
before the loss or validation metrics. It preserves matrix symmetry and rotation
covariance. It does not remove rotational modes from nonstationary geometries.
Targets retain their original values and the existing symmetrization policy.

Packed training retains directed off-diagonal blocks and node blocks. Assembly
forms `H = C + C^T`. The projection keeps the antisymmetric null space of `C`
and projects its symmetric part. Three Triton launches compute FP32 row means,
grand means, and corrected blocks. The backward uses the same self-adjoint
linear map. Neither pass assembles or pads a dense Hessian. The extra storage
contains corrected blocks, nine row values per atom, and nine values per geometry.
Dense validation uses block means and subtraction, without constructing `P`.

The checkpoint records `translation_projection`. Resume rejects a different
setting; evaluation restores the saved setting. Existing checkpoints default to
`False`. Set `TRANSLATION_PROJECTION=1` for the allocation wrapper. Inference
consumers must also apply this saved setting to the assembled Hessian.

`tools/evaluate_frozen_hip_translation.py` compares unchanged last checkpoints
before and after projection on the fixed validation cohort. It computes all four
mass-weighted, Eckart-projected metrics, Cartesian errors, and translation
residuals. Cartesian projection can change the mass-weighted projected metrics
when atomic masses differ; validation must measure that change.

The paired checkpoint comparison uses the same 10,027 fixed validation samples
and the last weights at 2,000,000 steps. Both checkpoints use weight decay
0.0001. Values below use mass weighting and Eckart projection. Compare the
paired baselines when assessing the correction.

| Checkpoint | Correction | abs cos(v1) | Eigenvalue MAE | Hessian MAE | Hessian RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| mae | None | 0.807613 | 0.076288 | 0.016940 | 0.035025 |
| mae | Cartesian PHP | 0.806996 | 0.076116 | 0.018110 | 0.035535 |
| block-rmse | None | 0.809268 | 0.073431 | 0.016643 | 0.034095 |
| block-rmse | Cartesian PHP | 0.809158 | 0.073384 | 0.018009 | 0.034779 |

Error units are eV/(Angstrom^2 amu). Mean translation residuals fall from
0.00612 and 0.00634 to approximately 1.47e-8. However, the mass-weighted,
Eckart-projected Hessian MAE increases by 6.9% and 8.2%. The postprocessing
correction therefore remains opt-in for existing checkpoints. A separate
zero-decay Block RMSE ablation trains with the projection in both the loss and
inference. Its target is 2,000,000 updates, with validation every 400,000 updates.
Production adoption depends on that ablation's validation results.

A warm microbenchmark compares the packed reference with the Triton projection,
including a simple backward objective. It shares a GPU with training; these are
not complete training-step measurements.

| Molecules and atoms | Reference forward/backward | Triton forward/backward |
| --- | ---: | ---: |
| 64 molecules, 40 atoms each | 4.65 ms | 0.61 ms |
| 3 molecules, 16/64/128 atoms | 15.07 ms | 0.87 ms |
| 1 molecule, 350 atoms | 109.74 ms | 0.61 ms |

Twenty-eight numerical and allocation tests pass, including ragged batches,
single atoms, 350 atoms, scale extremes, rotation covariance, exact-matrix
references, gradients, idempotence, and forwarding the allocation setting.

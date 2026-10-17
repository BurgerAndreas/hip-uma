# Frozen UMA-s-1.2 with an additional HIP head

This workflow freezes every pretrained UMA parameter. It trains the existing
equivariant `Hessian_Head` on dense UMA autograd Hessians. It retains the full
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

Use `--nproc_per_node=1` for a single-GPU run. The global batch limits stay the
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

The optional feature cache records exact FP32 frozen node features during the
first pass. Subsequent passes skip the backbone when a complete cache exists.
Cache namespaces include the frozen parameter hash and its TF32 setting.
Each cache file records its source shard hash. Incomplete caches stay in bounded
host memory and never become visible to readers. Validation recomputes its
features. Feature caching needs additional disk space and adds first-pass I/O.

Use `--head-precision bf16-mlp` to test BF16 in the scalar radial MLPs and
scalar gating MLPs. They return FP32 coefficients to the equivariant layers.
Parameters, optimizer state, geometry, rotations, equivariant projections,
Hessian assembly, and losses retain FP32. Standard layer normalization uses
FP32 under CUDA autocast. This mode preserves checkpoint parameter names.
Use `--head-precision bf16` to test autocast across the complete head.
The backbone keeps FP32 tensors. `--tf32` permits TensorFloat-32 matrix operations. Compare
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

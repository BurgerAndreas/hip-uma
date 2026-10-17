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

Run through your scheduler on a GPU compute node:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m fairchem.core.frozen_hip.train \
  --data "$DATA_DIR" --output "$RUN_DIR" \
  --workers 2 --pair-budget 32768 --max-batch 64 \
  --feature-cache "$FEATURE_CACHE_DIR"
```

All paths and scheduler settings come from the local environment. No scheduler
account, host, filesystem, or partition is required in tracked configuration.

The loader decompresses each shard once per pass. It groups mixed-size shards
into equal-size batches and bounds each batch by its quadratic pair count.
Different ranks read disjoint shard groups. Training excludes a deterministic
one percent of parent indices. Validation always uses those held-out indices.
The loss gives each molecular Hessian equal weight. DDP corrects for unequal
sample counts between ranks.

The optional feature cache records exact FP32 frozen node features during the
first pass. Subsequent passes skip the backbone when a complete cache exists.
Cache namespaces include the frozen parameter hash and its TF32 setting.
Each cache file records its source shard hash. Incomplete caches stay in bounded
host memory and never become visible to readers. Validation recomputes its
features. Feature caching needs additional disk space and adds first-pass I/O.

Use `--head-precision bf16` to test mixed precision in the head. The backbone
keeps FP32 tensors. `--tf32` permits TensorFloat-32 matrix operations. Compare
numerical errors and throughput with `tools/benchmark_frozen_hip.py` before
selecting these options. Large graphs use activation checkpointing according
to `--checkpoint-edges`.

Checkpoints contain the HIP head, optimizer state, step, frozen parameter hash,
and public model and data identifiers. The next invocation resumes `latest.pt`
automatically. It starts a new shuffled data pass. Checkpoints verify that the
backbone has no gradients and that every frozen parameter remains unchanged.
Scheduler time limits and termination signals trigger a final checkpoint.

Training logs report throughput, loss, and peak GPU memory. Validate convergence
on the held-out split before interpreting training throughput as model quality.

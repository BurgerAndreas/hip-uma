# Chunked molecular HIP inference

HIP can process all directed atom pairs in bounded chunks and return either a dense Hessian or symmetric block output.
The default chunk contains 65,536 candidate pairs.
This path supports one nonperiodic molecule on CUDA, with lmax=mmax=2, in evaluation mode.
It does not support training or graph parallelism.

```python
from fairchem.core.frozen_hip.model import FrozenHIP

model = FrozenHIP("cuda", backbone_graph_version=4).eval().requires_grad_(False)
# Supply one AtomicData batch on CUDA. Charge and spin retain UMA conventions.
hessian, directed_pairs = model.predict_hessian(data, chunk_edges=65536)
blocks, directed_pairs = model.predict_hessian(
    data, chunk_edges=65536, output_format="blocks"
)
```

Load a trained head checkpoint before scientific use.
Creating `FrozenHIP` alone initializes an untrained head.
The API retains FP32 backbone evaluation by default.
`precision="fp16"` selects reduced-precision frozen features.
Existing head precision helpers remain available; chunking does not change weights.

## Output representation

`HessianBlocks` stores:

- `indices`: int32 atom indices, shape `(2, K)`, with `i < j`.
- `values`: FP32 Cartesian blocks, shape `(K, 3, 3)`.
- `diagonal`: FP32 diagonal blocks, shape `(N, 3, 3)`.
- `row_ptr`: row offsets into the sorted pair list.
- `row_mean` and `grand_mean`: the implicit translation correction.

Both directed predictions contribute to each stored block.
The reverse block is its transpose.
The translation projection generally creates nonzero values at absent pairs.
Store that correction implicitly to preserve the complete projected operator:

`H_ij = raw_ij - row_mean[i] - row_mean[j].T + grand_mean`

The object provides `block(i, j)`, `matvec(vector)`, `save(path)`, and `load(path)`.
Serialization uses a versioned tensor dictionary and `weights_only=True` on load.
`to_dense()` is an explicit conversion; it requires memory for the complete matrix.
The benchmark returns blocks on the GPU without writing large matrices to disk.

All-to-all output remains quadratic.
Its upper-triangle blocks and indices use about 22*N*N bytes, plus linear-size arrays.
This is approximately 39 percent smaller than one complete FP32 dense Hessian.
Avoiding dense projection also removes its large temporary matrices.

## Distance cutoffs

Use `pair_cutoff=30.0` to filter pairs before allocating wide edge features.
The cutoff does not change the head's trained radial feature scale.
It changes head message passing and can change predictions inside the cutoff.
Validate scientific accuracy before applying a new cutoff to a trained head.
The implicit translation correction can introduce projected entries outside the cutoff.

Cutoff block output builds an uncapped spatial neighbor list once.
It bins atoms into occupied cells and tests the 27 adjacent cells.
Two GPU passes count and fill the canonical pairs, with no neighbor limit.
GPU kernels sort each row for block lookup and serialization.
The graph and sorting storage scale with `N+K`, where K counts retained pairs.
At fixed density and cutoff, this removes repeated all-candidate scans.
At large cutoffs, K can still approach `N*(N-1)/2`.

Each head pass reuses this graph and processes both directions of each pair.
The block writer combines both directions at a known output slot.
It avoids per-edge block searches and output atomics.
The message kernel accumulates into one FP32 node buffer.
It avoids a full-node temporary and full-node addition for each edge chunk.
These changes preserve weights, cutoffs, and the implicit translation correction.
Floating-point reduction order can change.

The scalar random-angle initialization remains quadratic.
It preserves the original CUDA random sequence and final RNG state.
The cutoff path gathers the exact retained angles, then frees the full buffer.
GPU angle storage uses one gather kernel.
CPU storage gathers retained angles on the host and transfers chunk slices.
During gathering, both full and retained angle buffers coexist.
Use `neighbor_list=False, direct_messages=False` for the original reference path.
Deterministic-algorithm mode keeps the reference backend dispatch.
Dense cutoff output and all-to-all output retain the original chunked graph builder.

## Large-system support

Backbone graph v4 supports up to 1,024 atoms per molecule.
The prediction API temporarily selects official graph v2 for larger inputs and restores the previous setting afterward.
CUDA edge kernels use 64-bit edge offsets to avoid overflow in large feature arrays.

`merged_backbone.merge_for_batch` provides the composition-specific expert merging used by the benchmark.
Its FP16 path retains FP32 accumulation in second-convolution output layers.
Merging changes the parameter representation, so use it only for the matching composition.

## Validation and measurements

GPU tests cover chunk boundaries, empty graphs, cutoffs, upper-block accumulation,
projection, matrix-vector products, individual blocks, and serialization round trips.
They also exercise the public API on a cached UMA-S-1.2 backbone.

On an H100, the development test at 438 atoms produced a relative Frobenius difference
of 2.14e-7 between block output and dense chunked output.
Cutoff tests at 20 and 60 angstrom also passed.
These tests use a randomly initialized performance head and do not establish model accuracy.

The earlier dense chunking and cutoff measurements are published in
[hip-uma-omol25-distill](https://github.com/BurgerAndreas/hip-uma-omol25-distill/tree/main/docs/benchmarks/hip_chunked_cutoffs_20261030).
The block-output sweep covers 19 systems from 438 to 29,292 atoms.

Package integration validation on H100: 21 tests passed.
This covers the public API, symmetric block kernels, serialization, translation
projection and gradients, and existing FP16 edge-precision behavior.

## Backbone chunking and large-system input limits

`FrozenHIP.predict_hessian` now chunks the frozen backbone edges as well as
the HIP all-pairs head. `backbone_chunk_edges=65536` is the default; pass `None`
to keep ordinary backbone dispatch. This changes no parameters or neighbor edges.
All edge messages are accumulated before each node update. Training dispatch
is unchanged, and temporary inference settings are restored after the call.

UMA's random charge embedding supports integer charges -100 through +100.
The HIP feature and expert-merging APIs reject unsupported charges before a
CUDA lookup. Do not clamp the charge or extend the table with untrained rows.
Prepare a physically distinct supported system, such as a protein with explicit
counterions, and label that benchmark separately from the bare protein.

`angle_storage="cpu"` copies the exact FP32 random-angle buffer to host RAM
and transfers slices back for each chunk. It preserves the original angle values
and includes transfer time in prediction time. It requires about `4*N*(N-1)`
bytes of host RAM, plus the process and model. The benchmark therefore requests
64 GiB host memory for systems near 110,000 atoms. It does not imply linear
total memory. The normal default keeps angles on the GPU.

Angle scaling now runs in place. The random-angle initialization no longer
allocates a second quadratic GPU tensor. All-to-all block output itself remains
quadratic and can exceed one H100 even after these changes.

## Cutoff profile and optimization measurements

The H100 control uses PDB 1G5A chain A, with 9,881 atoms and batch size one.
Each prediction recomputes backbone features and returns GPU block output.
It uses the same untrained one-layer head, FP16-edge arithmetic, TF32,
and 65,536-edge chunks. Timings are medians of three runs after one warmup.
Expert merging and the separate profiler call are outside the timed region.
The CPU-angle comparison keeps angle storage unchanged.

| Cutoff | Original, CPU angles | New, CPU angles | New, GPU angles |
| --- | ---: | ---: | ---: |
| 30 angstrom | 10.782 s | 5.397 s | 4.683 s |
| 60 angstrom | 13.020 s | 11.191 s | 9.921 s |

Total peak allocated GPU memory stays at 3.951 GiB in these completed tests.
The backbone sets that peak. The head alone does not always use less memory.
At 30 angstrom, its peak changes from 1.843 to 2.024 GiB with CPU angles.
The early retained pair list explains this additional storage.
GPU angles raise the head peak to 2.187 GiB at this cutoff.

The 30 angstrom profile reduces `aten::nonzero` calls from 5,967 to seven.
It removes 1,490 head segmented-scatter calls and 1,490 `bincount` calls.
The new spatial graph builds once. Both head passes use its 21,811,267 pairs.
The 60 angstrom graph contains 47,226,489 pairs, close to the complete graph.
This explains the smaller timing improvement at that cutoff.
These performance tests do not measure trained-head accuracy.

The timed source passes all 38 GPU tests.
They cover cutoff boundaries, exact angles and RNG state, both message backends,
block assembly, serialization, translation projection, and the previous backbone fixes.
All six larger-system measurements finished without an out-of-memory error.
These runs keep retained angles on the GPU; the original runs use CPU angles.
This comparison includes the neighbor list, direct accumulation, block writes,
and the change in angle storage. It does not isolate the neighbor list alone.
Inputs, retained pair counts, head initialization, and arithmetic match.

| Atoms | Cutoff | Original time | New time | Speedup | Original peak GPU | New peak GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31,768 | 30 angstrom | 126.25 s | 17.54 s | 7.20x | 9.66 GiB | 9.66 GiB |
| 31,768 | 60 angstrom | 145.58 s | 63.83 s | 2.28x | 13.82 GiB | 16.14 GiB |
| 102,089 | 30 angstrom | 1,896.91 s | 53.71 s | 35.32x | 39.33 GiB | 43.03 GiB |
| 102,089 | 60 angstrom | 2,139.80 s | 266.64 s | 8.02x | 53.48 GiB | 63.14 GiB |
| 109,942 | 30 angstrom | 2,290.48 s | 58.99 s | 38.83x | 45.57 GiB | 49.66 GiB |
| 109,942 | 60 angstrom | 2,579.33 s | 302.91 s | 8.52x | 60.80 GiB | 71.77 GiB |

GPU angle storage increases memory in several cases.
The full initial angle buffer also remains quadratic.
The results establish successful inference through 109,942 atoms at both cutoffs.
They do not establish a new maximum supported atom count.

A separate 9,881-atom comparison preserves exact pair indices and CUDA RNG state.
The relative Frobenius differences in raw off-diagonal blocks are 0.0269 percent
at 30 angstrom and 0.0127 percent at 60 angstrom.
All tested output fields differ by less than 0.027 percent.
These differences reflect the changed floating-point reduction order.

With CPU angles and all-to-all output at 9,881 atoms, direct message accumulation
reduces median prediction time from 12.404 to 10.910 seconds, or 1.14x.
Peak allocated GPU memory remains 3.951 GiB in that control.

## Size comparison beginning at 100 atoms

The comparison extends from 100 through 109,942 atoms on H100.
The original path stores angles on the CPU. The optimized path stores retained
angles on the GPU. This includes both the code changes and angle storage.
Each measurement includes fresh backbone features and block output, at batch
size one. One warmup precedes three timed calls; the table uses the median.
Expert merging and compilation setup remain outside the timed region.

The 100, 200, and 350 atom inputs are single OMol25 geometries.
All larger inputs are the existing prepared proteins.
Matched optimized CPU-angle controls are also stored in the raw results.

| Atoms | 30 angstrom speedup | 60 angstrom speedup | Extra GPU GiB, 30 / 60 angstrom |
| ---: | ---: | ---: | ---: |
| 100 | 1.01x | 1.00x | <0.01 / <0.01 |
| 200 | 1.02x | 1.01x | <0.01 / <0.01 |
| 350 | 1.04x | 1.03x | <0.01 / <0.01 |
| 596 | 1.08x | 1.08x | 0.00 / 0.00 |
| 1,231 | 1.11x | 1.10x | 0.00 / 0.00 |
| 2,977 | 1.22x | 1.15x | 0.00 / 0.00 |
| 4,968 | 1.37x | 1.19x | 0.00 / 0.00 |
| 9,881 | 2.30x | 1.31x | 0.00 / 0.00 |
| 31,768 | 7.20x | 2.28x | 0.00 / 2.32 |
| 102,089 | 35.32x | 8.02x | 3.69 / 9.66 |
| 109,942 | 38.83x | 8.52x | 4.09 / 10.97 |

At 100 atoms, the original and optimized predictions both take about 20 ms.
The gains below 600 atoms are small. With CPU angle storage unchanged,
the optimized path provides no speed gain at the four tested sizes below 600.
The small gains there come from keeping angles on the GPU.

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

The current graph builder scans all directed candidate pairs in bounded chunks.
Its wide activations are bounded, but its scalar rotation-roll buffer remains quadratic.
A spatial neighbor search is needed for linear graph-construction cost at fixed density.

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

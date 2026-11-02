# HIP precision measurements

Measured on one NVIDIA A100 80 GB GPU with a frozen UMA-s-1.2 backbone.
The head starts from the same trained parameters for each precision mode.
Numerical comparisons use identical FP32 frozen features and dense labels.
Throughput includes forward, backward, gradient clipping, and AdamW updates.
Each timing uses two warmup updates and six measured updates.

| Atoms | Molecules | TF32 speedup over FP32 | BF16 speedup over TF32 | TF32 output error | BF16 output error | BF16 gradient error | TF32 peak GB | BF16 peak GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 1.09x | 1.00x | 0.038% | 0.492% | 18.7% | 2.15 | 1.77 |
| 72 | 32 | 1.23x | 1.04x | 0.041% | 0.465% | 26.1% | 19.18 | 11.00 |
| 124 | 12 | 1.28x | 1.04x | 0.037% | 0.573% | 9.4% | 21.25 | 12.10 |
| 350 | 1 | 1.67x | 1.01x | 0.019% | 1.647% | 53.2% | 14.57 | 8.48 |

Output and gradient errors are relative L2 errors against strict FP32.
Speedups use end-to-end training throughput, including the frozen backbone.
Peak memory uses allocated GPU bytes and decimal GB.

BF16 across the head reduces memory, but gradient errors reach 53%.
Its end-to-end speed gain over TF32 stays below 5% in these cases.
BF16 only in scalar MLPs gives no consistent speed or memory gain.
The main training setting therefore keeps FP32 tensors and permits TF32 GEMMs.
Geometry, normalization, Hessian assembly, loss, parameters, and optimizer state
retain FP32. Smaller physical batches control memory through gradient accumulation.

Frozen backbone parameter hashes match before and after all measurements.
These numerical checks do not establish final validation accuracy equivalence.
Use held-out validation to assess a complete training run.

Reproduce with `tools/benchmark_hip_precision.py`.
Use `--checkpoint` to compare modes at the same head checkpoint.
Select representative molecule sizes with `--sizes`.

## Fused FP16 range packing

`--head-precision fp16-edge --fused-range-pack` selects an optional training path.
It retains FP32 parameters, matrix-product accumulations, reductions, and outputs.
The kernels reduce coalesced tiles, then pack normalized operands into FP16.
Already-FP16 inputs can bypass normalization and retain their original values.

The scalar convolution returns gates and messages separately.
Backward kernels read these gradients without a full concatenation buffer.
They compute FP32 bias sums during the range scan.
Complex convolutions also avoid concatenation buffers and fold weight gradients directly.
Parameter names and checkpoint tensor shapes remain unchanged.
Checkpoint metadata records the selected path for evaluation.

The path is optional. The H200 measurements below qualify its speed and accuracy in these tests.
These checks must include the complete head and activation checkpoint recomputation.
Short numerical tests do not establish final validation accuracy.

## Compiler artifacts across allocations

`--compiler-cache DIRECTORY` saves supported PyTorch compiler artifacts after a session.
The next process loads compatible artifacts before compiling the head.
The cache key includes source contents, runtime versions, GPU type, and head settings.
Missing, corrupt, incompatible, or shared writable artifacts cause a cache miss.
Normal compilation remains available, with compiler guards enabled.
Use a private directory. The cache contains executable compiler artifacts.

The segment wrapper accepts `FUSED_RANGE_PACK=1` and `COMPILER_CACHE_ROOT=DIRECTORY`.
Measure restart time separately from steady training time.

### H200 packing benchmark

The final dispatch uses the compiler path for contiguous FP32 operands.
Custom kernels handle strided operands and virtual concatenations.
Reproduce the component measurements with `tools/benchmark_hip_range_pack.py`.

The complete training test uses two conditioned head blocks with twice the standard width.
It retains the same checkpoint, optimizer, batch order, batch limits, and two loader workers.
Each replay runs 600 updates. Timing excludes the first 300 updates because compilation continues there.

| Measurement | Repeated baseline | Fused packing |
|---|---:|---:|
| Update time, matched updates 301–600 | 72.64 ms | 58.17 ms |
| Update time, matched updates 101–200 | 96.15 ms | 75.51 ms |
| Peak allocated GPU memory | 58.87 GB | 63.58 GB |
| Peak reserved GPU memory | 65.14 GB | 66.31 GB |

Throughput increases by 24.9% over the repeated baseline in the primary timing window.
These short windows do not establish throughput across the complete training distribution.
Peak allocated GPU memory increases by 4.71 GB.

The 100-update profile measures 20.41 ms of matrix products and 11.39 ms of layout copies per update.
Range packing and associated reductions now take about 8.58 ms, including casts and scaling.
The previous profile measured about 31.24 ms for range, bias, and scaling kernels.
The new trace contains about 919 kernels per update and about 16.5 ms of GPU gaps.
Remaining layout copies and kernel launch overhead therefore remain useful targets.
Profiler overhead is about 5.5% for the matched window.

A separate compiler-cache control uses the earlier packing variant.
Its session duration falls from 223.49 seconds with fresh compiler directories to 110.16 seconds with restored artifacts.
These durations include compilation and 600 updates. They do not measure steady training throughput.

See [the benchmark record](frozen_hip_range_pack_benchmark.json) for numerical differences and component timings.

The 39 focused GPU tests pass, including compiled backward and checkpoint recomputation.
A fixed 1,024-sample subset of the monitoring cohort also checks all four projected metrics.
These metrics use mass weighting and Eckart projection. Error units are eV/(Å²·amu).

| Metric | Repeated baseline | Optimized training endpoint |
|---|---:|---:|
| abs cos(v₁) | 0.848064 | 0.849662 |
| Eigenvalue MAE | 0.0277160 | 0.0276210 |
| Hessian MAE | 0.00295851 | 0.00295968 |
| Hessian RMSE | 0.00855436 | 0.00856813 |

The initial maximum-parameter gate failed, so qualification also used direct accuracy checks.
A two-sided endpoint v₁ check also failed because its score improved.
The final accuracy criterion checks degradation on trained endpoints and absolute agreement with identical weights.
It retains the original baseline-repeat tolerances. All eight checks pass.
The original probe used independent random local-frame rolls between heads.
Its identical-weight differences therefore include roll variation.
The paired-roll probe below supersedes that comparison for implementation qualification.
The accuracy probe uses eager evaluation; throughput measurements use compiled training.
Full-run accuracy still needs validation.

Cache artifacts also pass a GPU test across two nodes and two source directories.
The restored process records two graph-cache hits. Outputs and gradients match.

### Direct radial gradient reads

`--split-radial` keeps the three angular gradient groups separate during radial backward.
The kernels read each group directly, including reciprocal-edge node reductions.
This removes the full radial gradient concatenation buffer.
The forward values, FP16 operand storage, and FP32 reductions retain their existing definitions.
The dataset and Hessian targets retain their packed representation.

The flag requires `--head-precision fp16-edge`.
The segment wrapper accepts `SPLIT_RADIAL=1`.
Checkpoint metadata records the flag for resumed training and evaluation.
The combined configuration below qualifies this optional path for future training allocations.

The trainer also accepts `--python-switch-interval SECONDS` for controlled host scheduling tests.
The segment wrapper accepts `PYTHON_SWITCH_INTERVAL=SECONDS`.
The default retains the interpreter setting.
AMP reuses the finite result from its update helper and avoids two duplicate GPU scalar reads.
Overflow checks, loss scaling, retries, and checkpoint commit rules remain active.


### Layout copies and launch gaps

Use `FUSED_RANGE_PACK=1`, `SPLIT_RADIAL=1`, `FUSED_WIGNER=1`, and `PYTHON_SWITCH_INTERVAL=.001` together.
Keep `LOADER_WORKERS=2` and the existing private loader memory setup.

The next optimization keeps the radial backward groups separate through their consumers.
It avoids rebuilding the complete radial gradient tensor from the 3/4/2 angular groups.
Fused Wigner setup reduces the number of small rotation kernels.
A 1 ms Python switch interval helps the loader and GPU submission threads share the interpreter.
The trainer also removes two duplicate finite-result GPU reads per update.
It retains the original AMP checks and retries the same batch after overflow.

A matched 600-update replay compares the combined configuration with fused range packing alone.
The primary window uses accepted updates 301–600 and excludes compilation.
The reference takes 61.44 ms per update. The combined configuration takes 56.37 ms.
This increases throughput by 9.0%. Updates 101–200 improve by 4.3%.
Peak allocated and reserved GPU memory remain 63.58 GB and 66.31 GB.
A second replay includes the final duplicate finite-read cleanup in both paths.
It measures 61.91 ms for the reference and 56.93 ms for the combined path.
This confirms an 8.8% throughput increase on identical accepted batch sequences.
Final peak allocated memory is 63.58 GB. Reserved memory rises from 66.31 to 66.82 GB.
These short windows do not predict throughput for every molecule size.
Explicit cuBLAS selection and a larger heuristic cache show no consistent benefit.

The full CPU/CUDA trace reduces layout kernels with fused epilogues from 11.76 to 7.33 ms per update.
Those kernels also perform arithmetic and casts; this category is not pure memory copying.
Kernel count falls from about 919 to 778 per update.
Matrix products still take about 20.4 ms in that traced window.
CPU tracing increases host scheduling overhead, especially with the shorter switch interval.
Its apparent launch gaps therefore cannot establish the unprofiled improvement.
CPU use stays below the four allocated cores. Serial launch work can still delay GPU submission.

A second trace records CUDA activity alone for matched updates 301–400.
Its overhead is 4.0% for the reference and 2.6% for the combined path.
Layout kernels with fused epilogues fall from 12.57 to 7.94 ms per update.
Kernel count falls from 907 to 774 per update.
Gaps between kernels fall from 14.79 to 11.28 ms per update.
Conservative late host submissions explain 11.54 ms before and 8.96 ms after the change.
Kernel activity rises from 82.2% to 85.7% of the captured GPU time envelope.
Including GPU copies and memory initialization gives 82.5% before and 86.0% after.
The remaining device idle time is 11.05 ms per update.
These values measure trace activity, rather than sampled GPU utilization.
Both traces have complete launch correlations and no compilation events.
Matrix products still take 22.35 ms per update in this window.
Host submission, remaining layout work, and range reductions remain useful targets.


All 53 focused GPU tests pass, including compiled backward, checkpoint recomputation, and rotation edge cases.
A complete-head gradient check uses the same weights, batch, and random local-frame roll.
Eager gradients match exactly. Compiled relative L2 difference is 2.372e-5.
The repeated compiled baseline differs by 2.346e-5.
Both paths retain FP16 operand storage and FP32 products and reductions.

The paired accuracy probe uses 1,024 fixed monitoring samples and FP64 projected metrics.
It restores the same random local-frame roll before each head evaluation.
The combined trained endpoint and identical-weight comparisons pass the stated degradation limits.
Identical-weight Cartesian outputs differ by 0.0235% in relative L2 norm.
The radial-only trained endpoint fails three projected error limits despite matching eager gradients exactly.
That isolated endpoint is not accuracy-qualified. The combined package passes the separate check.
The short probes do not establish final training accuracy. Full validation remains necessary.

See [the layout benchmark record](frozen_hip_layout_benchmark.json) for paired metrics and gradient checks.

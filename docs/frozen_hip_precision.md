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

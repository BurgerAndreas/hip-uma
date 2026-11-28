# Large-System Size & Charge Scaling Findings (`HIP-UMA-S-1.2`)

Branch: `architecture-size-charge`. Reference checkpoint audited: `hip-uma-step4000000.pt` vs `UMA-S-1.2` centered finite differences (`FD_0.02_A`) and autograd VJP.

## 1. Empirical Findings

- **Chunked/dense & precision parity:** `chunked.predict` / `chunked.predict_hvp` matches dense `Hessian_Head` to `4.7e-6–6.0e-6` relative error. Switching backbone precision (`fp16` vs `fp32`) changes `5CSC_AB` (`13,212` atoms) curvature by `<0.5%`.
- **Two-cluster distance probe (`2.5–60 A`):** `blocks.0` cross-cluster message RMS does not decay, remaining `1.6–2.8` from `12 A` to `60 A` (peak `2.79` at `40 A`) due to unnormalized `index_add_` over a `0–100 A` Gaussian RBF (`col_norm = 8–18` for `d > 20 A`, untrained on OMol25 `N <= 350`).
- **5CSC_AB (`13,212` atoms, `UMA FD_0.02` ref curv = `+0.3975 eV/A^2`):**
  - `all_ch100` (mean deg `13,211`): `b1_x_rms = 54,014`, `rel_err = 16,426`, `curv = +14,925.7` (`569.6 s` A100).
  - `cut30_ch100` (mean deg `4,776`): `b1_x_rms = 7,398`, `rel_err = 34,142`, `curv = +74,300.0` (`205.0 s`).
  - `cut30_ch30` (envelope shrunk to `30 A`): `b1_x_rms = 2,202`, `rel_err = 21.64`, `curv = -0.5066` (`204.9 s`).
  - `msg9_read30_ch100` (message passing `9 A`, readout `30 A`): `b1_x_rms = 2,084`, `rel_err = 0.103`, `curv = +0.3792`.
  - `cut9_ch100_asr` (`9 A` + exact blockwise ASR): `rel_err = 0.083`, `cos = +0.9965`, `curv = +0.3998` (`+0.57%`, `10.3 s`).
  - `cut6_ch100_asr` (`6 A` + ASR): `rel_err = 0.084`, `cos = +0.9965`, `curv = +0.3973` (`-0.05%`, `3.4 s`).
- **5CSC_AB 10-mode collective subspace (Mode 0 hinge + Modes 1–9 ANM, `0.035–0.939 eV/A^2`):**
  - `cut12` (raw): `10x10` subspace eigenvalue rel err `44.04%` (Mode 1: `-84.0%`).
  - `cut12_asr`: `10x10` subspace eigenvalue rel err `2.80%` (Mode 1: `-1.5%`).
  - `cut9_asr`: `10x10` subspace eigenvalue rel err `3.26%` (Mode 1: `-1.7%`).
- **Out-of-distribution global net charge (`1CFD`, `N = 2,262`, `q = -24`):** `hip_charge_spin` extrapolates out of distribution (`cut9_ch100_asr` `rel_err = 6.20`, `b0_msg_rms = 6,308` vs `~890` on other proteins). Clamping `charge_clamp = 4.0` removes the blowup.
- **In-distribution `DFT1078` (`matched936`, clean `n = 932`, `N = 4–175`, `diam = 3.0–102.7 A`, `50.1% > 12 A`):**
  - All `932`: `all` vs `cut12` `hess_mae = 0.06750` vs `0.06750`, `freq_mae = 19.97` vs `20.01 cm^-1`; post-hoc `cut12_asr` `freq_mae = 20.30 cm^-1`.
  - Largest tail (`diam > 18 A`, `n = 76`): `all` `freq_mae = 9.23 cm^-1`, `low<200 = 13.49 cm^-1`, `low3_ov = 0.7398` vs `cut12_asr` `freq_mae = 8.99 cm^-1`, `low<200 = 12.61 cm^-1`, `low3_ov = 0.7623`.

## 2. Implemented Changes (`architecture-size-charge`)

- **Inference (`fairchem.core.frozen_hip.chunked.predict_hvp`):** Defaults to `cutoff=9.0`, `asr=True`, `charge_clamp=4.0`, keeping `cutoff_hessian=100.0` fixed; `radius_pbc_version=4` automatic fallback for `N > 1024`.
- **Retraining architecture (`configs/frozen_hip/expanded_size_charge_cut20_block_rmse.json`):**
  - `head_message_cutoff = 12.0` (smooth `12 A` envelope on `blocks.0–1`).
  - `head_readout_cutoff = 20.0` (smooth `20 A` envelope on `edge_readout`).
  - `head_structural_asr = true` ($H_{ii} = -\sum_{j \neq i} H_{ij}$ by construction).
  - `head_charge_spin_mode = "intensive_bounded"` (`Q/N`, `(S-1)/N` with `tanh`-bounded gains).

# TODO

## Active SLURM Jobs

- `4481453` `horm_uma_ef_gpu` is `PENDING` on `batch_singlenode`.
  - Runs `scripts/submit_horm_uma_energy_force_gpu.sh` from this repo.
  - Nodes are down, drained, or reserved for higher-priority partitions.
  - Submitted at `2026-05-25T06:20:16`, with an estimated start shown by SLURM of `2026-05-25T09:33:58`.

## Next

- Recheck `4481453`. If it stays pending, cancel it and resubmit on `batch_block1` or another available higher-priority GPU partition.
- After it starts, tail the log until the first checkpoint and evaluation complete.

## HIP Hessian Prediction for UMA

### Goal

Implement learned HIP Hessian prediction for UMA/eSCN-MD. The target architecture should follow the Equiformer HIP path in `../hip`, building a Hessian graph, predicting off-diagonal 3x3 edge blocks and diagonal 3x3 node blocks from equivariant l=0,1,2 features, converting those irreps to Cartesian matrices, and scattering the blocks into a symmetric flattened Hessian with length `sum_b (3 * N_b)^2`.

### Reference Code

- Equiformer HIP graph, index, and assembly utilities live in `../hip/nets/equiformer_v2/hessian_pred_utils.py`.
  - `irreps_to_cartesian_matrix()` at line 31.
  - `blocks3x3_to_hessian()` at line 351.
  - `blocks3x3_to_hessian_loops()` at line 380.
  - `add_hessian_graph_batch()` at line 409.
- Equiformer HIP model path lives in `../hip/nets/equiformer_v2/equiformer_v2_oc20.py`.
  - Hessian projection modules around lines 563-568.
  - Hessian forward path around lines 876-976.
- Equiformer HIP training and loss references.
  - Hessian-only trainable module names in `../hip/hip/training_module.py` around lines 83-86.
  - Hessian loss handling in `../hip/hip/training_module.py` around line 442.
  - Optional eigenspectrum metrics and losses in `../hip/hip/loss_functions.py`.
- UMA model references.
  - Backbone and head definitions in `src/fairchem/core/models/uma/escn_md.py`.
  - `Edgewise` and `eSCNMD_Block` in `src/fairchem/core/models/uma/escn_md_block.py`.
  - `SO3_Linear` in `src/fairchem/core/models/uma/nn/so3_layers.py`.
  - Wigner and rotation helpers in `src/fairchem/core/models/uma/common/rotation.py`, and quaternion helpers in `src/fairchem/core/models/uma/common/quaternion/`.
- UMA data and training references.
  - `AtomicData.__cat_dim__`, `AtomicData.__inc__`, and `atomicdata_list_to_batch()` in `src/fairchem/core/datasets/atomic_data.py`.
  - `Task` and `compute_loss()` in `src/fairchem/core/units/mlip_unit/mlip_unit.py`.
  - `MTCollater` in `src/fairchem/core/datasets/collaters/mt_collater.py`.
  - HORM converter in `tools/convert_hip_lmdb_to_atomic_lmdb.py`, which already writes flattened `hessian` tensors.
  - Current energy and force configs in `configs/uma/training_release/horm_uma_energy_force.yaml` and `configs/uma/training_release/horm_uma_test.yaml`.

### Data Pieces

- Keep Hessian targets as flattened per-system tensors.
  - Source sample shape can be `[3N, 3N]`, `[N, 3, N, 3]`, or flattened.
  - Dataset output should be `hessian.reshape(-1)` with length `(3N)^2`.
  - Batched target should be concatenated into one vector with length `sum_b (3N_b)^2`.
- Add Hessian validation to `AtomicData`.
  - Optional key `hessian`.
  - Dtype must match `pos`.
  - For single samples, `hessian.numel() == (3 * natoms.item()) ** 2`.
  - For batches, `hessian.numel() == sum((3 * natoms) ** 2)`.
- Add Hessian batching metadata.
  - `ptr_1d_hessian = [0, cumsum((3 * natoms) ** 2)]`.
  - Optionally `hessian_nentries = (3 * natoms) ** 2`.
  - This can live in the collater, a transform, or a helper called by the Hessian graph builder.
- Ensure `exclude_keys: ["hessian"]` remains only in energy and force configs. Hessian configs must retain the target.
- Update `tools/convert_hip_lmdb_to_atomic_lmdb.py` only if needed after validation is added, since it already writes `out["hessian"] = hessian.reshape(-1)`.

### Hessian Graph and Assembly Utilities

- Port and adapt `../hip/nets/equiformer_v2/hessian_pred_utils.py` into a UMA-local module, likely `src/fairchem/core/models/uma/hessian_pred_utils.py`.
- Needed utilities include the following.
  - `get_cartesian_wigner_3j_basis()`.
  - `irreps_to_cartesian_matrix()`.
  - Fully connected Hessian graph builder for molecular HORM use.
  - Radius and cutoff Hessian graph builder using `fairchem.core.graph.compute.generate_graph()`.
  - `add_hessian_graph_batch()` for `AtomicData`.
  - Off-diagonal scatter indices `message_idx_ij` and `message_idx_ji`.
  - Diagonal scatter indices `diag_ij`, `diag_ji`, and `node_transpose_idx`.
  - `blocks3x3_to_hessian()` and slow reference `blocks3x3_to_hessian_loops()`.
- UMA adaptation points.
  - Use `AtomicData` keys `pos`, `batch`, `natoms`, `cell`, and `pbc`.
  - Handle `batch_full` and graph-parallel fields by rejecting graph parallel in the first implementation.
  - Ensure edge indices use global atom indices while scatter indices use per-system local offsets into flattened Hessian segments.
  - Keep `fully_connected_hessian: true` as the initial HORM setting to match `../hip`.

### UMA Edge-Message Access

- Current UMA `Edgewise.forward()` in `src/fairchem/core/models/uma/escn_md_block.py` computes edge messages and immediately scatters them back to nodes.
- Learned HIP needs edge-level equivariant messages before the final edge-to-node scatter.
- Add one of these APIs.
  - `Edgewise.forward_messages(...)`, returning post-`SO2_Convolution` and activation edge embeddings in the edge frame or global frame as needed.
  - A Hessian-specific edge message module that reuses the same internal layers and returns edge messages.
  - A Hessian-specific `eSCNMD_Block` variant that can optionally return messages from the final Hessian graph pass.
- Verify that the returned edge representation is compatible with l=0,1,2 slicing and Cartesian conversion.
  - Expected edge message tensor shape is `[E_hessian, (lmax + 1)^2, C]`.
  - The Hessian path needs only the first 9 coefficients for l=0,1,2.

### Direct Hessian Head

- Add a UMA head class, such as `Hessian_Head`, in `src/fairchem/core/models/uma/escn_md.py` or a new module imported there.
- Constructor should receive the backbone and configure the following components.
  - `num_layers_hessian`.
  - `cutoff_hessian`.
  - `fully_connected_hessian`.
  - Hessian radial basis and distance expansion.
  - Hessian-specific `eSCNMD_Block` stack.
  - Edge l012 projection `SO3_Linear(backbone.sphere_channels or message_channels, 1, lmax=2)`.
  - Node l012 projection `SO3_Linear(backbone.sphere_channels, 1, lmax=2)`.
  - Cached Cartesian Wigner-3j basis buffer.
- Forward path steps.
  - Start from final UMA node embeddings `emb["node_embedding"]`.
  - Build the Hessian graph and scatter indices.
  - Compute Hessian edge distance vectors, distances, Wigner matrices, and inverse Wigner matrices with the same backbone rotation and backend path.
  - Build Hessian edge scalar features from distance expansion plus source and target element embeddings.
  - Run optional Hessian-specific message-passing layers on a copy of the node embedding.
  - Obtain edge equivariant messages on the Hessian graph.
  - Slice l012 edge features and l012 node features.
  - Project both to one 9-component irrep vector per edge and node.
  - Convert each 9-component irrep vector to a 3x3 Cartesian block.
  - Assemble the flattened Hessian via `blocks3x3_to_hessian()`.
  - Return `{"hessian": hessian_flat}` under the configured task name.
- Symmetry convention should match Equiformer HIP.
  - Off-diagonal block `(i, j)` receives the predicted edge block.
  - Block `(j, i)` receives its transpose.
  - Diagonal node blocks are added with their transpose.

### Hydra, Task, and Loss Plumbing

- Add a Hessian task definition.
  - `name: hessian`.
  - `property: hessian`.
  - Likely `level: system`, with special handling for the variable flattened length.
  - Normalizer initially identity unless a HORM Hessian scale is computed.
  - Metrics initially `mae` and `mse`, or custom Hessian metrics.
- Extend `compute_loss()` in `src/fairchem/core/units/mlip_unit/mlip_unit.py` for `task.property == "hessian"`.
  - Do not reshape predictions to `[batch_size, -1]`.
  - Use `batch.ptr_1d_hessian` to slice per-system segments.
  - Build dataset masks at the system level and expand them to Hessian-entry masks.
  - Compare `predictions[task.name]["hessian"].view(-1)` with `batch[task.name].view(-1)`.
  - Ignore `train_on_free_atoms` and `fixed` for Hessian loss.
- Add Hessian-specific losses if the `DDPMTLoss` assumptions are too rigid.
  - `HessianMAELoss`.
  - `HessianMSELoss`.
  - Optional batched per-structure averaging so larger molecules do not outweigh smaller ones.
- Extend `compute_metrics()` similarly for Hessian MAE and MSE.
- Check `MTCollater._add_missing_attr()` for mixed-task batches.
  - Variable Hessian target dimensions do not fit the current fixed `out_spec` pattern.
  - Either restrict initial Hessian training to HORM-only batches or add a Hessian special case for missing attributes.

### Config Pieces

- Add a HIP HORM config, likely beside `configs/uma/training_release/horm_uma_energy_force.yaml`.
- Config should include these fields.
  - `exclude_keys: []` or an exclude list that does not include `hessian`.
  - `heads.hessian.module: fairchem.core.models.uma.escn_md.Hessian_Head`.
  - Energy and force heads as needed for multitask training.
  - Hessian task with coefficient comparable to Equiformer HIP (`../hip/configs/train.yaml` uses `hessian_loss_weight: 10.0`, with comments noting HORM used a Hessian weight around 4).
  - `fully_connected_hessian: true`.
  - `cutoff_hessian: 100.0`.
  - `num_layers_hessian: 1` initially.
- Decide whether Hessian head training should support full model training, freezing the backbone to train only Hessian modules (analogous to `train_hessian_only` in `../hip`), or both.
- Add a small test config using `configs/uma/training_release/backbone/K2L2_test.yaml` or another tiny backbone.

### Inference and API Pieces

- Ensure MLIP inference can request and return a `hessian` task from the Hydra model.
- Decide output shape for public inference, noting that internal training output is a flattened concatenated vector while single-system calculator output should probably reshape to `[3N, 3N]`.
- Add calculator and unit support where needed.
  - `src/fairchem/core/units/mlip_unit/api/inference.py`.
  - `src/fairchem/core/calculate/ase_calculator.py`.
  - Any benchmark or singlepoint runner that filters requested properties.

### Tests

- Utility tests.
  - `irreps_to_cartesian_matrix()` shape, dtype, and device.
  - Index-add assembly equals `blocks3x3_to_hessian_loops()`.
  - Fully connected Hessian graph has `N * (N - 1)` directed edges per system.
  - Variable-natoms batch creates matching `ptr_1d_hessian` and scatter indices.
- Data and loss tests.
  - `AtomicData` accepts valid flattened Hessians and rejects wrong lengths.
  - `atomicdata_list_to_batch()` concatenates Hessian targets accurately.
  - Hessian loss slices per sample accurately for different `natoms`.
  - Hessian dataset masking works for HORM-only and mixed batches if mixed batches are supported.
- Model tests.
  - `Hessian_Head` forward works for one small molecule.
  - `Hessian_Head` forward works for a batch with different `natoms`.
  - Output length is `sum_b (3 * N_b)^2`.
  - Output is symmetric when reshaped per sample.
  - Rotation equivariance satisfies `H_rot ~= (I kron R) H (I kron R)^T`.
- Training smoke tests.
  - Overfit 1 to 4 HORM molecules and confirm Hessian MAE decreases.
  - Compare predicted Hessian symmetry and eigenvalue sanity on held-out HORM samples.
  - Run one short GPU job with the tiny backbone before scaling.

### Recommended Implementation Order

1. Add Hessian target validation and batching metadata.
2. Port Hessian graph, index, and assembly utilities into UMA and test them.
3. Add Hessian loss and metric special cases for flattened variable-length targets.
4. Add edge-message access to UMA `Edgewise` or a Hessian-specific message module.
5. Implement `Hessian_Head`.
6. Add HIP HORM configs and keep `hessian` targets in those dataloaders.
7. Add forward, symmetry, equivariance, and loss tests.
8. Run a tiny overfit, then scale to TS1x and HORM training.

# TODO

## Active SLURM Jobs

- `4481453` `horm_uma_ef_gpu`: `PENDING` on `batch_singlenode`.
  - Runs `scripts/submit_horm_uma_energy_force_gpu.sh` from this repo.
  - Reason: nodes are down/drained or reserved for higher-priority partitions.
  - Submitted: `2026-05-25T06:20:16`; estimated start shown by SLURM: `2026-05-25T09:33:58`.

## Next

- Recheck `4481453`; if it stays pending, cancel it and resubmit on `batch_block1` or another available higher-priority GPU partition.
- After it starts, tail the log until the first checkpoint/eval completes.

## HIP Hessian Prediction for UMA

### Goal

Implement direct learned HIP Hessian prediction for UMA/eSCN-MD. The target architecture should follow the Equiformer HIP path in `../hip`: build a Hessian graph, predict off-diagonal 3x3 edge blocks and diagonal 3x3 node blocks from equivariant l=0,1,2 features, convert those irreps to Cartesian matrices, then scatter the blocks into a symmetric flattened Hessian with length `sum_b (3 * N_b)^2`.

### Reference Code

- Equiformer HIP graph/index/assembly utilities: `../hip/nets/equiformer_v2/hessian_pred_utils.py`
  - `irreps_to_cartesian_matrix()` at line 31.
  - `blocks3x3_to_hessian()` at line 351.
  - `blocks3x3_to_hessian_loops()` at line 380.
  - `add_hessian_graph_batch()` at line 409.
- Equiformer HIP model path: `../hip/nets/equiformer_v2/equiformer_v2_oc20.py`
  - Hessian projection modules around lines 563-568.
  - Hessian forward path around lines 876-976.
- Equiformer HIP training/loss references:
  - Hessian-only trainable module names in `../hip/hip/training_module.py` around lines 83-86.
  - Hessian loss handling in `../hip/hip/training_module.py` around line 442.
  - Optional eigenspectrum metrics/losses in `../hip/hip/loss_functions.py`.
- UMA model references:
  - Backbone/head definitions in `src/fairchem/core/models/uma/escn_md.py`.
  - `Edgewise` and `eSCNMD_Block` in `src/fairchem/core/models/uma/escn_md_block.py`.
  - `SO3_Linear` in `src/fairchem/core/models/uma/nn/so3_layers.py`.
  - Wigner/rotation helpers in `src/fairchem/core/models/uma/common/rotation.py` and quaternion helpers in `src/fairchem/core/models/uma/common/quaternion/`.
- UMA data/training references:
  - `AtomicData.__cat_dim__`, `AtomicData.__inc__`, and `atomicdata_list_to_batch()` in `src/fairchem/core/datasets/atomic_data.py`.
  - `Task` and `compute_loss()` in `src/fairchem/core/units/mlip_unit/mlip_unit.py`.
  - `MTCollater` in `src/fairchem/core/datasets/collaters/mt_collater.py`.
  - HORM converter in `tools/convert_hip_lmdb_to_atomic_lmdb.py`, which already writes flattened `hessian` tensors.
  - Current energy/force configs in `configs/uma/training_release/horm_uma_energy_force.yaml` and `configs/uma/training_release/horm_uma_test.yaml`.

### Data Pieces

- Keep Hessian targets as flattened per-system tensors:
  - source sample shape can be `[3N, 3N]`, `[N, 3, N, 3]`, or flattened;
  - dataset output should be `hessian.reshape(-1)` with length `(3N)^2`;
  - batched target should be concatenated to one vector with length `sum_b (3N_b)^2`.
- Add Hessian validation to `AtomicData`:
  - optional key `hessian`;
  - dtype must match `pos`;
  - for single samples, `hessian.numel() == (3 * natoms.item()) ** 2`;
  - for batches, `hessian.numel() == sum((3 * natoms) ** 2)`.
- Add Hessian batching metadata:
  - `ptr_1d_hessian = [0, cumsum((3 * natoms) ** 2)]`;
  - optionally `hessian_nentries = (3 * natoms) ** 2`;
  - this can live in the collater, a transform, or a helper called by the Hessian graph builder.
- Make sure `exclude_keys: ["hessian"]` remains only in energy/force-only configs. Hessian configs must retain the target.
- Update `tools/convert_hip_lmdb_to_atomic_lmdb.py` only if needed after validation is added; it already writes `out["hessian"] = hessian.reshape(-1)`.

### Hessian Graph and Assembly Utilities

- Port/adapt `../hip/nets/equiformer_v2/hessian_pred_utils.py` into a UMA-local module, likely `src/fairchem/core/models/uma/hessian_pred_utils.py`.
- Needed utilities:
  - `get_cartesian_wigner_3j_basis()`;
  - `irreps_to_cartesian_matrix()`;
  - fully connected Hessian graph builder for molecular HORM use;
  - radius/cutoff Hessian graph builder using `fairchem.core.graph.compute.generate_graph()`;
  - `add_hessian_graph_batch()` for `AtomicData`;
  - off-diagonal scatter indices `message_idx_ij` and `message_idx_ji`;
  - diagonal scatter indices `diag_ij`, `diag_ji`, and `node_transpose_idx`;
  - `blocks3x3_to_hessian()` and slow reference `blocks3x3_to_hessian_loops()`.
- UMA adaptation points:
  - use `AtomicData` keys: `pos`, `batch`, `natoms`, `cell`, `pbc`;
  - handle `batch_full`/graph-parallel fields by explicitly rejecting graph parallel for the first implementation;
  - ensure edge indices use global atom indices while scatter indices use per-system local offsets into flattened Hessian segments;
  - keep `fully_connected_hessian: true` as the initial HORM setting to match `../hip`.

### UMA Edge-Message Access

- Current UMA `Edgewise.forward()` in `src/fairchem/core/models/uma/escn_md_block.py` computes edge messages and immediately scatters them back to nodes.
- Direct HIP needs edge-level equivariant messages before the final edge-to-node scatter.
- Add one of these APIs:
  - `Edgewise.forward_messages(...)`, returning post-`SO2_Convolution`/activation edge embeddings in the edge frame or global frame as needed;
  - a Hessian-specific edge message module that reuses the same internal layers but returns edge messages;
  - a Hessian-specific `eSCNMD_Block` variant that can optionally return messages from the final Hessian graph pass.
- Verify the returned edge representation is compatible with l=0,1,2 slicing and Cartesian conversion:
  - expected edge message tensor shape should be `[E_hessian, (lmax + 1)^2, C]`;
  - the Hessian path only needs the first 9 coefficients for l=0,1,2.

### Direct Hessian Head

- Add a UMA head class, e.g. `Hessian_Head`, in `src/fairchem/core/models/uma/escn_md.py` or a new module imported there.
- Constructor should receive the backbone and configure:
  - `num_layers_hessian`;
  - `cutoff_hessian`;
  - `fully_connected_hessian`;
  - Hessian radial basis/distance expansion;
  - Hessian-specific `eSCNMD_Block` stack;
  - edge l012 projection: `SO3_Linear(backbone.sphere_channels or message_channels, 1, lmax=2)`;
  - node l012 projection: `SO3_Linear(backbone.sphere_channels, 1, lmax=2)`;
  - cached Cartesian Wigner-3j basis buffer.
- Forward path:
  - start from final UMA node embeddings `emb["node_embedding"]`;
  - build Hessian graph and scatter indices;
  - compute Hessian edge distance vectors, distances, Wigner matrices, and inverse Wigner matrices with the same backbone rotation/backend path;
  - build Hessian edge scalar features from distance expansion plus source/target element embeddings;
  - run optional Hessian-specific message-passing layers on a copy of the node embedding;
  - obtain edge equivariant messages on the Hessian graph;
  - slice l012 edge features and l012 node features;
  - project both to one 9-component irrep vector per edge/node;
  - convert each 9-component irrep vector to a 3x3 Cartesian block;
  - assemble flattened Hessian via `blocks3x3_to_hessian()`;
  - return `{"hessian": hessian_flat}` under the configured task name.
- Symmetry convention should match Equiformer HIP:
  - off-diagonal block `(i, j)` receives the predicted edge block;
  - block `(j, i)` receives its transpose;
  - diagonal node blocks are added with their transpose.

### Hydra/Task/Loss Plumbing

- Add a Hessian task definition:
  - `name: hessian`;
  - `property: hessian`;
  - likely `level: system`, but special-case the variable flattened length;
  - normalizer initially identity unless a HORM Hessian scale is computed;
  - metrics initially `mae`/`mse` or custom Hessian metrics.
- Extend `compute_loss()` in `src/fairchem/core/units/mlip_unit/mlip_unit.py` for `task.property == "hessian"`:
  - do not reshape predictions to `[batch_size, -1]`;
  - use `batch.ptr_1d_hessian` to slice per-system segments;
  - build dataset masks at the system level and expand them to Hessian-entry masks;
  - compare `predictions[task.name]["hessian"].view(-1)` with `batch[task.name].view(-1)`;
  - ignore `train_on_free_atoms`/`fixed` for Hessian loss.
- Add Hessian-specific losses if the generic `DDPMTLoss` assumptions are too rigid:
  - `HessianMAELoss`;
  - `HessianMSELoss`;
  - optional batched per-structure averaging to avoid larger molecules dominating.
- Extend `compute_metrics()` similarly for Hessian MAE/MSE.
- Check `MTCollater._add_missing_attr()` for mixed-task batches:
  - variable Hessian target dimensions do not fit the current fixed `out_spec` pattern;
  - either restrict initial Hessian training to HORM-only batches or add a Hessian special-case for missing attributes.

### Config Pieces

- Add a direct HIP HORM config, likely beside `configs/uma/training_release/horm_uma_energy_force.yaml`.
- Config should include:
  - `exclude_keys: []` or an exclude list that does not include `hessian`;
  - `heads.hessian.module: fairchem.core.models.uma.escn_md.Hessian_Head`;
  - energy and force heads as needed for multitask training;
  - Hessian task with coefficient comparable to Equiformer HIP (`../hip/configs/train.yaml` uses `hessian_loss_weight: 10.0`, with comments noting HORM used Hessian weight around 4);
  - `fully_connected_hessian: true`;
  - `cutoff_hessian: 100.0`;
  - `num_layers_hessian: 1` initially.
- Decide whether Hessian head training should support:
  - full model training;
  - freezing the backbone and training only Hessian modules, analogous to `train_hessian_only` in `../hip`.
- Add a small test config using `configs/uma/training_release/backbone/K2L2_test.yaml` or another tiny backbone.

### Inference/API Pieces

- Ensure MLIP inference can request and return a `hessian` task from the Hydra model.
- Decide output shape for public inference:
  - internal training output: flattened concatenated vector;
  - single-system calculator output should probably reshape to `[3N, 3N]`.
- Add calculator/unit support where needed:
  - `src/fairchem/core/units/mlip_unit/api/inference.py`;
  - `src/fairchem/core/calculate/ase_calculator.py`;
  - any benchmark/singlepoint runner that filters requested properties.

### Tests

- Utility tests:
  - `irreps_to_cartesian_matrix()` shape, dtype, and device;
  - index-add assembly equals `blocks3x3_to_hessian_loops()`;
  - fully connected Hessian graph has `N * (N - 1)` directed edges per system;
  - variable-natoms batch creates correct `ptr_1d_hessian` and scatter indices.
- Data/loss tests:
  - `AtomicData` accepts valid flattened Hessians and rejects wrong lengths;
  - `atomicdata_list_to_batch()` concatenates Hessian targets correctly;
  - Hessian loss slices per sample correctly for different `natoms`;
  - Hessian dataset masking works for HORM-only and mixed batches if mixed batches are supported.
- Model tests:
  - `Hessian_Head` forward works for one small molecule;
  - `Hessian_Head` forward works for a batch with different `natoms`;
  - output length is `sum_b (3 * N_b)^2`;
  - output is symmetric when reshaped per sample;
  - rotation equivariance: `H_rot ~= (I kron R) H (I kron R)^T`.
- Training smoke tests:
  - overfit 1-4 HORM molecules and confirm Hessian MAE decreases;
  - compare predicted Hessian symmetry/eigenvalue sanity on held-out HORM samples;
  - run one short GPU job with the tiny backbone before scaling.

### Recommended Implementation Order

1. Add Hessian target validation and batching metadata.
2. Port Hessian graph/index/assembly utilities into UMA and test them.
3. Add Hessian loss/metric special-cases for flattened variable-length targets.
4. Add edge-message access to UMA `Edgewise` or a Hessian-specific message module.
5. Implement `Hessian_Head`.
6. Add direct HIP HORM configs and keep `hessian` targets in those dataloaders.
7. Add forward, symmetry, equivariance, and loss tests.
8. Run tiny overfit, then scale to TS1x/HORM training.

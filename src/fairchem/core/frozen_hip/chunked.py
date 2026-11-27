"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Edge-chunked HIP Hessian inference for large molecules and subset regions.

The dense head keeps every per-edge activation of the Hessian graph at the same
time. This path evaluates the same layers on edge chunks and sums the node
messages before each node update, so peak memory scales with ``chunk_edges``
and the Hessian output, not with the edge count. It reuses the head modules
directly, so input_projection, charge/spin conditioning, block_so2 and
projected_readout follow the dense computation. It applies no translation
projection.
"""

from __future__ import annotations

import math

import torch

from fairchem.core.models.uma.hessian_pred_utils import irreps_to_cartesian_matrix
from fairchem.core.models.uma.outputs import get_l_component_range


def pair_list(pos, cutoff=None, rows=1024):
    """
    Build the directed Hessian graph without an N x N distance matrix.

    Keep the order of the dense fully connected graph: target-major, then
    source. With a cutoff, keep pairs with squared distance <= cutoff**2, as
    the FairChem radius graph does (no neighbor cap).

    Args:
        pos: (n, 3) coordinates.
        cutoff: Pair cutoff in Angstrom, or None for all pairs.
        rows: Number of target atoms in one distance block.

    Returns:
        (2, E) long tensor of (source, target) indices.
    """
    n = pos.shape[0]
    parts = []
    every = torch.arange(n, device=pos.device)
    for begin in range(0, n, rows):
        target = every[begin : begin + rows]
        d2 = (pos[None, :, :] - pos[target, None, :]).square().sum(-1)
        keep = target[:, None] != every[None, :]
        if cutoff is not None:
            keep &= d2 <= cutoff * cutoff
        t, s = keep.nonzero(as_tuple=True)
        parts.append(torch.stack((s, target[t])))
        del d2, keep
    return torch.cat(parts, dim=1)


def roll_angles(head, edges, device):
    """
    Draw the random Wigner roll angles once, as the dense path does.

    The draw matches the dense call (one ``torch.rand`` over all edges), so a
    dense and a chunked run from the same RNG state use the same angles. The
    Euler-angle Wigner path cannot take angles; it returns None (fresh angles
    per chunk; the result then differs from dense by FP32 roundoff only).
    """
    if getattr(head, "fused_wigner", False):
        return torch.rand(edges, device=device) * (2 * torch.pi)
    if head.backbone.use_quaternion_wigner:
        return torch.rand(edges, dtype=torch.float32, device=device) * 2 * math.pi
    return None


def rotations(head, vec, dist, gamma=None, cutoff=None):
    """
    Return Wigner matrices; reuse ``gamma`` when the backend accepts it.
    """
    backbone = head.backbone
    if getattr(head, "fused_wigner", False):
        from fairchem.core.frozen_hip.wigner import frozen_wigner

        return frozen_wigner(head, vec, dist, gamma=gamma, cutoff=cutoff)
    if gamma is not None and backbone.use_quaternion_wigner:
        from fairchem.core.models.uma.common.quaternion.wigner_d_hybrid import (
            axis_angle_wigner_hybrid,
        )

        data = backbone.wigner_data
        return axis_angle_wigner_hybrid(
            vec,
            backbone.lmax,
            gamma=gamma.to(vec.dtype),
            coeffs=data.coeffs,
            U_blocks=data.U_blocks,
            custom_kernels=data.custom_kernels,
        )
    return backbone._get_rotmat_and_wigner(vec)


def edge_features(
    head,
    data,
    edge_index,
    gamma=None,
    *,
    cutoff_scale=None,
    distance_expansion=None,
    use_invr_envelope=False,
):
    """
    Compute radial features and rotations for one edge chunk.

    Mirror ``Hessian_Head._hessian_edge_features`` for an explicit edge list.
    """
    vec = data.pos[edge_index[0]] - data.pos[edge_index[1]]
    dist = vec.norm(dim=-1)
    fused = getattr(head, "fused_wigner", False) and not use_invr_envelope
    scale = head.cutoff_hessian if cutoff_scale is None else float(cutoff_scale)
    if fused:
        from fairchem.core.frozen_hip.wigner import frozen_wigner

        wigner, wigner_inv = frozen_wigner(head, vec, dist, gamma=gamma, cutoff=scale)
    else:
        wigner, wigner_inv = rotations(head, vec, dist, gamma, cutoff=scale)
    backbone = head.backbone
    coefficient_index = (
        backbone.coefficient_index if backbone.mmax != backbone.lmax else None
    )
    wigner, wigner_inv = head.backend.prepare_wigner(
        wigner, wigner_inv, backbone.mappingReduced, coefficient_index
    )
    if not fused:
        if use_invr_envelope:
            from fairchem.core.models.uma.escn_md import inverse_r3_envelope

            envelope = inverse_r3_envelope(
                dist, getattr(head, "readout_envelope_scale", 6.0)
            ).reshape(-1, 1, 1)
        else:
            envelope = backbone.envelope((dist / scale).clamp(max=1.0)).reshape(
                -1, 1, 1
            )
        wigner_inv = wigner_inv * envelope
    numbers = data.get("atomic_numbers_full", data.atomic_numbers)
    dist_fn = (
        head.distance_expansion_hessian
        if distance_expansion is None
        else distance_expansion
    )
    x_edge = torch.cat(
        (
            dist_fn(dist),
            backbone.source_embedding(numbers[edge_index[0]]),
            backbone.target_embedding(numbers[edge_index[1]]),
        ),
        dim=1,
    )
    return x_edge, wigner, wigner_inv


def _check(head, data):
    from fairchem.core.common import gp_utils

    if gp_utils.initialized():
        raise NotImplementedError("Chunked Hessians do not support graph parallelism")
    if head.training or torch.is_grad_enabled():
        raise ValueError("Chunked Hessians support frozen inference only")
    if data.natoms.numel() != 1:
        raise ValueError("Chunked Hessians require one structure per call")
    if bool(data.pbc.any()) and getattr(head, "use_pbc_hessian", False):
        raise ValueError("Chunked Hessians require a nonperiodic Hessian graph")
    if getattr(head, "requires_mole_routing", True) or hasattr(head, "hip_mole_router"):
        raise ValueError("Chunked Hessians require a head without MoLE routing")


def _clamped_charge(charge, charge_clamp):
    if charge_clamp is None:
        return charge
    limit = float(charge_clamp)
    return charge.clamp(min=-limit, max=limit)


def _run_message_blocks(head, data, emb, msg_edges, gamma_msg, chunk_edges, charge_clamp=None):
    n = data.pos.shape[0]
    total = msg_edges.shape[1]
    x = emb["node_embedding"].clone()
    if hasattr(head, "hip_charge_spin"):
        x = head.hip_charge_spin(
            x, _clamped_charge(data.charge, charge_clamp), data.spin, data.batch
        )
    x = head.input_projection(x)
    msg_scale = getattr(head, "message_cutoff_hessian", None)
    msg_dist_fn = (
        getattr(head, "distance_expansion_message", None) if msg_scale is not None else None
    )
    for block in head.blocks:
        residual = x
        normalized = block.norm_1(x)
        messages = None
        for begin in range(0, total, chunk_edges):
            edges = msg_edges[:, begin : begin + chunk_edges]
            roll = None if gamma_msg is None else gamma_msg[begin : begin + chunk_edges]
            x_edge, wigner, wigner_inv = edge_features(
                head,
                data,
                edges,
                roll,
                cutoff_scale=msg_scale,
                distance_expansion=msg_dist_fn,
            )
            part = block.edge_wise.forward_chunk(
                normalized, n, x_edge, edges, wigner, wigner_inv
            )
            messages = part if messages is None else messages.add_(part)
            del part, x_edge, wigner, wigner_inv
        x = residual if messages is None else messages + residual
        x = x + block.atom_wise(block.norm_2(x))
        del messages, normalized, residual

    x = head.norm(x)
    node_irreps = head.node_projection(
        get_l_component_range(x, l_min=0, l_max=2)
    ).squeeze(-1)
    basis = head.cartesian_wigner_3j_basis.to(
        device=node_irreps.device, dtype=node_irreps.dtype
    )
    node_blocks = irreps_to_cartesian_matrix(node_irreps, basis)
    return x, node_blocks, basis


def _readout_chunk(head, x, basis, n, edges, x_edge, wigner, wigner_inv):
    if getattr(head, "projected_readout", False):
        irreps = head.projected_edge_irreps(x, x_edge, edges, wigner, wigner_inv)
    else:
        features = head.edge_readout.forward_messages(
            x, x_edge, edges, wigner, wigner_inv, total_atoms_across_gp_ranks=n
        )
        irreps = head.edge_projection(
            get_l_component_range(features, l_min=0, l_max=2)
        ).squeeze(-1)
    return irreps_to_cartesian_matrix(irreps, basis)


@torch.no_grad()
def predict(
    head,
    data,
    emb,
    chunk_edges=65536,
    cutoff=None,
    output_format="dense",
    edge_index=None,
    *,
    msg_cutoff=None,
    msg_edge_index=None,
    asr=False,
    charge_clamp=None,
):
    """
    Predict one Hessian with edge chunks.

    Rotations use one random roll angle per edge, drawn once and reused in
    every layer and the readout, as the dense path does. From the same RNG
    state and edge order the result equals the dense head up to summation
    order; otherwise it differs by the dense head's own roll noise.

    Args:
        head: A ``Hessian_Head`` in eval mode (no MoLE routing).
        data: AtomicData with one structure; ``pos`` holds the Hessian atoms.
        emb: Backbone output; ``emb["node_embedding"]`` has one row per atom of
            ``data``. For a subset region, pass the subset rows of the
            full-system embedding.
        chunk_edges: Directed edges per chunk.
        cutoff: Pair cutoff in Angstrom for readout (and message passing unless
            ``msg_cutoff`` is set), or None for all pairs. The radial basis and
            envelope always keep the trained ``head.cutoff_hessian`` scale.
        output_format: "dense" returns a (3n, 3n) FP32 tensor. "blocks" returns
            (edge_index, edge_blocks, node_blocks) with H = sum over edges of
            C on block (s, t) and C.T on block (t, s), plus N + N.T on (i, i).
        edge_index: Optional precomputed (2, E) readout graph; overrides ``cutoff``.
        msg_cutoff: Optional separate pair cutoff for message-passing blocks
            (e.g. 9.0 A to keep node degrees in-distribution while reading out
            longer-range pairs). Defaults to ``cutoff``.
        msg_edge_index: Optional precomputed (2, E_msg) message-passing graph.
        asr: If True, enforce exact blockwise acoustic sum rule by replacing
            diagonal blocks with ``-sum_{j!=i} H_sym_ij``.
        charge_clamp: Optional symmetric clamp ``[-c, +c]`` on ``data.charge``
            passed to ``head.hip_charge_spin`` to prevent OOD extrapolation on
            highly charged macromolecules.

    Returns:
        The Hessian in the requested format.
    """
    _check(head, data)
    if chunk_edges < 1:
        raise ValueError("chunk_edges must be positive")
    if output_format not in ("dense", "blocks"):
        raise ValueError("output_format must be dense or blocks")
    asr = bool(asr or getattr(head, "structural_asr", False))
    n = data.pos.shape[0]
    if edge_index is None:
        edge_index = pair_list(data.pos, cutoff)
    total = edge_index.shape[1]
    starts = range(0, total, chunk_edges)
    gamma = roll_angles(head, total, data.pos.device)

    eff_msg_cutoff = (
        msg_cutoff
        if msg_cutoff is not None
        else getattr(head, "message_cutoff_hessian", None)
    )
    if msg_edge_index is not None:
        msg_edges = msg_edge_index
        gamma_msg = roll_angles(head, msg_edges.shape[1], data.pos.device)
    elif eff_msg_cutoff is not None and eff_msg_cutoff != cutoff:
        msg_edges = pair_list(data.pos, eff_msg_cutoff)
        gamma_msg = roll_angles(head, msg_edges.shape[1], data.pos.device)
    else:
        msg_edges = edge_index
        gamma_msg = gamma

    use_invr_readout = (
        getattr(head, "readout_radial_mode", "gaussian") == "inverse_distance"
    )

    def chunks():
        for begin in starts:
            edges = edge_index[:, begin : begin + chunk_edges]
            roll = None if gamma is None else gamma[begin : begin + chunk_edges]
            yield (
                edges,
                *edge_features(
                    head, data, edges, roll, use_invr_envelope=use_invr_readout
                ),
            )

    x, node_blocks, basis = _run_message_blocks(
        head, data, emb, msg_edges, gamma_msg, chunk_edges, charge_clamp=charge_clamp
    )

    if output_format == "blocks":
        blocks = node_blocks.new_empty((total, 3, 3))
        row_sum = (
            torch.zeros((n, 3, 3), device=x.device, dtype=torch.float64) if asr else None
        )
        for begin, (edges, *f) in zip(starts, chunks()):
            part = _readout_chunk(head, x, basis, n, edges, *f)
            blocks[begin : begin + edges.shape[1]] = part
            if row_sum is not None:
                part64 = part.double()
                row_sum.index_add_(0, edges[0], part64)
                row_sum.index_add_(0, edges[1], part64.transpose(-1, -2))
        if row_sum is not None:
            # Diagonal assembly uses N + N.T, so N = -0.5 * row_sum yields H_ii = -row_sum.
            node_blocks = (-0.5 * row_sum).to(dtype=node_blocks.dtype)
        return edge_index, blocks, node_blocks

    hessian = node_blocks.new_zeros((3 * n) * (3 * n))
    width = 3 * n
    c = torch.arange(3, device=x.device)
    ci, cj = c.view(1, 3, 1), c.view(1, 1, 3)
    row_sum = (
        torch.zeros((n, 3, 3), device=x.device, dtype=torch.float64) if asr else None
    )
    for edges, *f in chunks():
        blocks = _readout_chunk(head, x, basis, n, edges, *f)
        s = edges[0].view(-1, 1, 1)
        t = edges[1].view(-1, 1, 1)
        hessian.index_add_(
            0, ((s * 3 + ci) * width + t * 3 + cj).reshape(-1), blocks.reshape(-1)
        )
        hessian.index_add_(
            0,
            ((t * 3 + ci) * width + s * 3 + cj).reshape(-1),
            blocks.transpose(-1, -2).reshape(-1),
        )
        if row_sum is not None:
            b64 = blocks.double()
            row_sum.index_add_(0, edges[0], b64)
            row_sum.index_add_(0, edges[1], b64.transpose(-1, -2))
        del blocks, f
    hessian = hessian.view(n, 3, n, 3)
    ids = torch.arange(n, device=x.device)
    if row_sum is not None:
        hessian[ids, :, ids, :] += (-row_sum).to(dtype=hessian.dtype)
    else:
        hessian[ids, :, ids, :] += node_blocks + node_blocks.transpose(-1, -2)
    return hessian.view(width, width)


@torch.no_grad()
def predict_hvp(
    head,
    data,
    emb,
    direction,
    *,
    chunk_edges=65536,
    cutoff=9.0,
    msg_cutoff=None,
    asr=True,
    charge_clamp=4.0,
    edge_index=None,
    msg_edge_index=None,
):
    """
    Streamed Hessian-vector product H @ direction in O(N + chunk_edges) memory.

    Defaults to the large-system inference settings (``cutoff=9.0`` A,
    ``asr=True``, ``charge_clamp=4.0``) that keep message-passing degrees
    in-distribution and eliminate coherent acoustic sum-rule drift without
    materializing (3N, 3N) or (E, 3, 3) tensors.

    Args:
        head: A ``Hessian_Head`` in eval mode (no MoLE routing).
        data: AtomicData with one structure.
        emb: Backbone output dictionary containing ``node_embedding``.
        direction: Array or tensor of shape ``(n, 3)`` or ``(3 * n,)``.
        chunk_edges: Directed edges per chunk.
        cutoff: Pair cutoff in Angstrom (default 9.0), or None for all pairs.
        msg_cutoff: Optional separate message-passing cutoff; defaults to ``cutoff``.
        asr: If True (default), enforce exact blockwise ASR via
            ``(H_asr v)_i = sum_{j!=i} H_sym_ij (v_j - v_i)``.
        charge_clamp: Optional symmetric clamp on ``data.charge`` passed to
            ``head.hip_charge_spin`` (default 4.0). Pass None to disable.
        edge_index: Optional precomputed (2, E) readout graph.
        msg_edge_index: Optional precomputed (2, E_msg) message-passing graph.

    Returns:
        FP64 tensor of shape ``(n, 3)`` on ``data.pos.device``.
    """
    _check(head, data)
    if chunk_edges < 1:
        raise ValueError("chunk_edges must be positive")
    asr = bool(asr or getattr(head, "structural_asr", False))
    n = data.pos.shape[0]
    v = torch.as_tensor(direction, device=data.pos.device, dtype=torch.float64).reshape(n, 3)

    if edge_index is None:
        edge_index = pair_list(data.pos, cutoff)
    total = edge_index.shape[1]
    gamma = roll_angles(head, total, data.pos.device)

    eff_msg_cutoff = (
        msg_cutoff
        if msg_cutoff is not None
        else getattr(head, "message_cutoff_hessian", None)
    )
    if msg_edge_index is not None:
        msg_edges = msg_edge_index
        gamma_msg = roll_angles(head, msg_edges.shape[1], data.pos.device)
    elif eff_msg_cutoff is not None and eff_msg_cutoff != cutoff:
        msg_edges = pair_list(data.pos, eff_msg_cutoff)
        gamma_msg = roll_angles(head, msg_edges.shape[1], data.pos.device)
    else:
        msg_edges = edge_index
        gamma_msg = gamma

    x, node_blocks, basis = _run_message_blocks(
        head, data, emb, msg_edges, gamma_msg, chunk_edges, charge_clamp=charge_clamp
    )

    if asr:
        hv = torch.zeros((n, 3), device=data.pos.device, dtype=torch.float64)
    else:
        d64 = node_blocks.double()
        hv = ((d64 + d64.transpose(1, 2)) @ v[..., None]).squeeze(-1)

    use_invr_readout = (
        getattr(head, "readout_radial_mode", "gaussian") == "inverse_distance"
    )
    for begin in range(0, total, chunk_edges):
        edges = edge_index[:, begin : begin + chunk_edges]
        roll = None if gamma is None else gamma[begin : begin + chunk_edges]
        x_edge, wigner, wigner_inv = edge_features(
            head, data, edges, roll, use_invr_envelope=use_invr_readout
        )
        blocks = _readout_chunk(head, x, basis, n, edges, x_edge, wigner, wigner_inv).double()
        s, t = edges[0], edges[1]
        if asr:
            dv_st = (v[t] - v[s])[..., None]
            hv.index_add_(0, s, (blocks @ dv_st).squeeze(-1))
            hv.index_add_(0, t, (blocks.transpose(1, 2) @ (-dv_st)).squeeze(-1))
        else:
            hv.index_add_(0, s, (blocks @ v[t, :, None]).squeeze(-1))
            hv.index_add_(0, t, (blocks.transpose(1, 2) @ v[s, :, None]).squeeze(-1))
        del x_edge, wigner, wigner_inv, blocks
    return hv

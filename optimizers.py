# Copyright (c) 2026, Markus Knoche. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any

import torch


def _init_random(sample: torch.Tensor, K: int, N_init: int) -> torch.Tensor:
    N, S, C = sample.shape
    device = sample.device
    idx = torch.randint(S, (N_init, N, K), device=device)
    sample_exp = sample.unsqueeze(0).expand(N_init, -1, -1, -1)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)
    init = sample_exp.gather(2, idx_exp)
    return init.detach().clone()


@torch.enable_grad()
def optimize_fde_adam(
    sample: torch.Tensor,
    K: int,
    adam_steps: int = 300,
    adam_lr: float = 0.2,
    N_init: int = 10,
) -> dict[str, Any]:
    """Optimize trajectory endpoints for the minFDE objective with Adam.

    For each agent, this function selects ``K`` endpoints from sampled endpoint
    positions by minimizing the average distance from each sample to its nearest
    endpoint. Multiple random initializations are optimized in parallel, and the best
    initialization is selected independently for each agent.

    Args:
        sample:
            Sampled endpoint positions with shape ``[N, S, 2]``, where ``N`` is the
            number of agents and ``S`` is the number of samples.
        K:
            Number of endpoints to optimize per agent.
        adam_steps:
            Number of Adam optimization steps.
        adam_lr:
            Learning rate used by the Adam optimizer.
        N_init:
            Number of random initializations optimized in parallel.

    Returns:
        A dictionary containing ``endpoints`` with shape ``[N, K, 2]``.
    """
    N, _, _ = sample.shape
    sample_exp = sample.unsqueeze(0).expand(N_init, -1, -1, -1)
    endpoints_param = torch.nn.Parameter(_init_random(sample, K, N_init))

    params = [endpoints_param]

    opt = torch.optim.Adam(params, lr=adam_lr)

    for _ in range(adam_steps):
        opt.zero_grad()
        diff = sample_exp.unsqueeze(3) - endpoints_param.unsqueeze(2)
        sq_d = diff.pow(2).sum(-1)
        d = sq_d.clip_(1e-6).sqrt()
        loss_pts, _ = d.min(dim=3)

        loss = loss_pts.mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        diff = sample_exp.unsqueeze(3) - endpoints_param.unsqueeze(2)
        sq_d = diff.pow(2).sum(-1)
        d = sq_d.clamp_min(1e-6).sqrt()

        min_dists, _ = d.min(dim=3)
        loss_pts = min_dists

        per_init_n = loss_pts.mean(dim=2)

        best_idx = per_init_n.argmin(dim=0)
        best_endpoints = endpoints_param.detach()[best_idx, torch.arange(N)]

        return {'endpoints': best_endpoints}


def optimize_map_rectangles(
    positions: torch.Tensor,
    headings: torch.Tensor,
    velocities: torch.Tensor,
    K: int,
    thresh_long: float,
) -> dict[str, Any]:
    """Greedily place oriented rectangles to cover sampled endpoints for mAP.

    For each agent, this function places up to ``K`` oriented rectangles over the
    sampled positions. At each iteration, it selects the sample position and heading
    whose rectangle covers the largest number of currently uncovered samples. Rectangle
    size is derived from ``thresh_long`` and adjusted based on the current agent
    velocity.

    Args:
        positions:
            Sampled positions with shape ``[N, S, 2]``, where ``N`` is the number of
            agents and ``S`` is the number of samples.
        headings:
            Sampled headings with shape ``[N, S]``.
        velocities:
            Current agent velocities with shape ``[N]``.
        K:
            Maximum number of rectangles to place per agent.
        thresh_long:
            Longitudinal threshold used to determine the rectangle length.

    Returns:
        A dictionary containing:
            ``centers``:
                Rectangle centers with shape ``[N, K, 2]``.
            ``angles``:
                Rectangle orientations with shape ``[N, K]``.
            ``counts``:
                Number of samples covered by each rectangle, with shape``[N, K]``.
            ``lengths``:
                Velocity-adjusted half-lengths with shape ``[N]``.
            ``widths``:
                Velocity-adjusted half-widths with shape ``[N]``.
            ``pi``:
                Placeholder confidence field, currently set to ``None``.
    """
    device = positions.device
    N, S, _ = positions.shape

    endpoints = torch.empty(N, K, 2, device=device, dtype=positions.dtype)
    angles = torch.empty(N, K, device=device, dtype=positions.dtype)
    counts = torch.zeros(N, K, device=device, dtype=positions.dtype)
    lengths = torch.empty(N, device=device, dtype=positions.dtype)
    widths = torch.empty(N, device=device, dtype=positions.dtype)

    slope = (thresh_long - thresh_long / 2) / (11.0 - 1.4)
    half_L = torch.clamp(
        thresh_long / 2 + (velocities - 1.4) * slope,
        min=thresh_long / 2,
        max=thresh_long,
    ).to(positions.dtype)
    half_W = half_L / 2.0

    lengths[:] = half_L
    widths[:] = half_W

    half_L_b = half_L.view(N, 1, 1)
    half_W_b = half_W.view(N, 1, 1)

    diff = positions[:, None, :, :] - positions[:, :, None, :]
    dx = diff[..., 0]
    dy = diff[..., 1]

    c = headings.cos().unsqueeze(-1)
    s = headings.sin().unsqueeze(-1)

    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    inside = (local_x.abs() <= half_L_b) & (local_y.abs() <= half_W_b)
    del diff, dx, dy, local_x, local_y

    arangeN = torch.arange(N, device=device)

    active = torch.ones(N, S, dtype=positions.dtype, device=device)

    for k in range(K):
        cand_scores = (inside.to(positions.dtype) * active[:, :, None]).sum(dim=1)

        best_idx = cand_scores.argmax(dim=1)
        best_scores = cand_scores[arangeN, best_idx]

        endpoints[:, k] = positions[arangeN, best_idx]
        angles[:, k] = headings[arangeN, best_idx]
        counts[:, k] = best_scores

        if best_scores.max() <= 0:
            endpoints[:, k:] = 0
            counts[:, k:] = 0
            break

        covered = inside[arangeN, :, best_idx]

        active[covered] = 0.0

    return {
        'endpoints': endpoints,
        'angles': angles,
        'counts': counts,
        'lengths': lengths,
        'widths': widths,
        'pi': None,
    }

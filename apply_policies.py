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

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from optimizers import optimize_map_rectangles, optimize_fde_adam
from utils import pack_npy, unpack_npy


K = 6
WAYMO_RECT_LENGTHS = (2, 3.6, 6)


@dataclass(frozen=True)
class DatasetConfig:
    steps_in_pkl: tuple[int, ...]
    total_length: int
    subsample: int
    policies: tuple[str, ...]
    rect_lengths: tuple[float, ...] = WAYMO_RECT_LENGTHS


DATASET_CONFIGS = {
    'waymo': DatasetConfig(
        steps_in_pkl=(29, 49, 79),
        total_length=80,
        subsample=5,
        policies=('fde_adam', 'mAP_rectangles'),
        rect_lengths=WAYMO_RECT_LENGTHS,
    ),
}


def _build_full_trajectories(
    *,
    endpoints_steps: torch.Tensor,
    steps_in_pkl: tuple[int, ...],
    total_length: int,
) -> torch.Tensor:
    N, K_, _, _ = endpoints_steps.shape

    out = torch.zeros(
        (N, K_, total_length, 2),
        dtype=endpoints_steps.dtype,
        device=endpoints_steps.device,
    )

    for m, t in enumerate(steps_in_pkl):
        out[:, :, t, :] = endpoints_steps[:, :, m, :]

    return out


def _stack_scene_samples(
    scene_samples: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    pos_all = []
    head_all = []

    for agent in scene_samples.values():
        pos = unpack_npy(agent['samples_pos'])
        pos_all.append(torch.from_numpy(pos).to(torch.float32))

        if 'samples_head' in agent:
            head = unpack_npy(agent['samples_head'])
            head_all.append(torch.from_numpy(head).to(torch.float32))

    pos_all = torch.stack(pos_all)

    if len(head_all) > 0:
        head_all = torch.stack(head_all)
        if head_all.shape[-1] == 1:
            head_all = head_all.squeeze(-1)
        if head_all.shape != pos_all.shape[:3]:
            raise ValueError(
                f'Shape of pos and head samples are incompatible: {pos_all.shape} and {head_all.shape}.'
            )

        return pos_all, head_all

    return pos_all, None


def _stack_naive_forecasts(
    scene_samples: dict[str, dict[str, Any]],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    naive_pi = []
    naive_pos_steps = []

    for entry in scene_samples.values():
        if 'naive_pos' not in entry or 'naive_pi' not in entry:
            return None, None
        pos = unpack_npy(entry['naive_pos'])
        pi = unpack_npy(entry['naive_pi'])

        naive_pi.append(torch.from_numpy(pi).to(torch.float32))
        naive_pos_steps.append(torch.from_numpy(pos).to(torch.float32))

    return torch.stack(naive_pi), torch.stack(naive_pos_steps)


def _get_current_velocities(
    scene_gt: dict[str, dict[str, Any]],
    agent_ids: list[str],
) -> torch.Tensor:
    cur_vels = []

    for aid in agent_ids:
        a_gt = scene_gt[aid]
        vel = unpack_npy(a_gt['cur_vel'])
        cur_vels.append(torch.from_numpy(vel))

    cur_vels = torch.stack(cur_vels)
    cur_vels = torch.norm(cur_vels, dim=-1)

    return cur_vels.to(torch.float32)


def _pack_predictions(
    *,
    agent_ids: list[str],
    pi: torch.Tensor,
    pos_full: torch.Tensor,
    subsample: int,
) -> dict[str, dict[str, bytes]]:
    pos_sub = pos_full[:, :, subsample - 1 :: subsample]

    out = {}

    assert len(pi) == len(agent_ids)
    assert len(pos_sub) == len(agent_ids)
    assert len(agent_ids) > 0

    for agent_pi, agent_pos, agent_id in zip(pi, pos_sub, agent_ids):
        out[agent_id] = {
            'pi': pack_npy(agent_pi.cpu().numpy()),
            'pos': pack_npy(agent_pos.cpu().numpy()),
        }

    return out


def apply_policies(
    *,
    samples_db: dict[str, dict[str, Any]],
    gt_db: dict[str, dict[str, Any]],
    dataset: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Apply all configured TraDiE policies for a dataset.

    For each scenario and agent, this function reads horizon endpoint samples from
    ``samples_db`` and converts them into policy-specific endpoints.

    The returned predictions use the same nested layout as the input data:
    ``scenario_id -> agent_id -> prediction fields``. Each predicted agent stores packed
    arrays for ``pos`` and ``pi``.

    Args:
        samples_db:
            Nested sample dictionary loaded from the forecasting-method pickle.
        gt_db:
            Nested ground-truth dictionary loaded from the compact GT pickle.
        dataset:
            Dataset name for selecting the policy configuration.

    Returns:
        A tuple ``(predictions_naive, predictions_by_opt)``. ``predictions_naive``
        contains optional naive baseline predictions if they are present in the sample
        data. ``predictions_by_opt`` maps each policy name to its nested prediction
        dictionary.
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f'Dataset {dataset!r} not known. Valid datasets: {sorted(DATASET_CONFIGS)}'
        )

    cfg = DATASET_CONFIGS[dataset]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Dataset={dataset} device={device}')
    print(f'policies={list(cfg.policies)}')
    print(f'steps_in_pkl={cfg.steps_in_pkl}')
    print(f'total_length={cfg.total_length}')
    print(f'subsample={cfg.subsample}')

    torch.set_grad_enabled(False)

    predictions_naive = {}
    predictions_by_opt = {opt: {} for opt in cfg.policies}

    scene_keys = list(gt_db.keys())

    for scenario_id in tqdm(
        scene_keys,
        smoothing=50 / len(scene_keys),
        desc='Applying policies',
    ):
        scene_gt = gt_db[scenario_id]
        scene_samples = samples_db[scenario_id]

        agent_ids = list(scene_samples.keys())

        pos_all, head_all = _stack_scene_samples(scene_samples)

        pi_naive, pos_naive_steps = _stack_naive_forecasts(
            scene_samples=scene_samples,
        )

        if pi_naive is not None and pos_naive_steps is not None:
            pos_naive_full = _build_full_trajectories(
                endpoints_steps=pos_naive_steps,
                steps_in_pkl=cfg.steps_in_pkl,
                total_length=cfg.total_length,
            )

            predictions_naive[scenario_id] = _pack_predictions(
                agent_ids=agent_ids,
                pi=pi_naive,
                pos_full=pos_naive_full,
                subsample=cfg.subsample,
            )

        N, S, _, _ = pos_all.shape

        for policy in cfg.policies:
            if policy == 'fde_adam':
                endpoints = []

                for col in range(len(cfg.steps_in_pkl)):
                    pos_step = pos_all[:, :, col, :].to(device)

                    res = optimize_fde_adam(
                        pos_step,
                        K=K,
                    )

                    endpoints_step = res['endpoints'].detach().cpu()  # (N, K, 2)
                    endpoints.append(endpoints_step)

                endpoints = torch.stack(endpoints, dim=2)  # (N, K, M, 2)

                pi = torch.full((N, K), 1.0 / K, dtype=torch.float32)

            elif policy == 'mAP_rectangles':
                velocities = _get_current_velocities(
                    scene_gt=scene_gt,
                    agent_ids=agent_ids,
                ).to(device)
                if head_all is None:
                    raise ValueError(f'mAP_rectangles requires heads')

                endpoints = []

                for col in range(len(cfg.steps_in_pkl)):
                    pos_step = pos_all[:, :, col, :].to(device)  # (N, S, 2)
                    head_step = head_all[:, :, col].to(device)
                    rect_len = cfg.rect_lengths[col]

                    res = optimize_map_rectangles(
                        pos_step,
                        headings=head_step,
                        velocities=velocities,
                        K=K,
                        thresh_long=rect_len,
                    )

                    endpoints_step = res['endpoints'].detach().cpu()  # (N, K, 2)
                    endpoints.append(endpoints_step)

                endpoints = torch.stack(endpoints, dim=2)  # (N, K, M, 2)

                pi = res['counts'].detach().cpu() / float(S)

            else:
                raise RuntimeError(f'Unknown policies: {policy}')

            trajs = _build_full_trajectories(
                endpoints_steps=endpoints,
                steps_in_pkl=cfg.steps_in_pkl,
                total_length=cfg.total_length,
            )

            predictions_by_opt[policy][scenario_id] = _pack_predictions(
                agent_ids=agent_ids,
                pi=pi,
                pos_full=trajs,
                subsample=cfg.subsample,
            )

    return predictions_naive, predictions_by_opt


def main() -> None:
    """Apply TraDiE policies to sampled forecasts and write prediction pickles.

    This script loads a sample pickle produced by a forecasting method and a compact
    ground-truth pickle. It then applies all policies configured for the selected
    dataset and writes one prediction pickle per policy to the output directory. If the
    sample data contains a naive forecast baseline, an additional naive prediction
    pickle is written.

    Command-line arguments:
        --gt_path:
            Path to the compact ground-truth pickle.
        --sample_path:
            Path to the sample pickle produced by the forecasting method.
        --pred_dir:
            Directory where the optimized prediction pickles should be written.
        --dataset:
            Dataset name selecting the corresponding TraDiE policy configuration.

    Outputs:
        Prediction pickles in ``--pred_dir``. Policy outputs are written as
        ``<sample_stem>-<policy>.pkl``. If available, naive baseline predictions
        are written as ``<sample_stem>-naive.pkl``.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--sample_path', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--dataset', default='waymo', choices=DATASET_CONFIGS.keys())

    args = parser.parse_args()

    gt_path = Path(args.gt_path)
    sample_path = Path(args.sample_path)
    pred_dir = Path(args.pred_dir)

    output_stem = sample_path.stem

    print(f'Using samples from: {sample_path}')
    print(f'Using GT from: {gt_path}')
    print(f'Writing predictions to: {pred_dir}')

    print('Loading pickles...')
    with open(sample_path, 'rb') as f:
        samples_db = pickle.load(f)

    with open(gt_path, 'rb') as f:
        gt_db = pickle.load(f)

    print()
    print(f'Loaded {len(samples_db)} scenarios from samples')
    print(f'Loaded {len(gt_db)} scenarios from GT')

    if set(gt_db) != set(samples_db):
        raise ValueError('Scenarios in gt and samples differ!')

    predictions_naive, predictions_by_opt = apply_policies(
        samples_db=samples_db,
        gt_db=gt_db,
        dataset=args.dataset,
    )

    os.makedirs(pred_dir, exist_ok=True)

    if len(predictions_naive) > 0:
        naive_path = pred_dir / f'{output_stem}-naive.pkl'

        with open(naive_path, 'wb') as f:
            pickle.dump(predictions_naive, f)

        print(f'Wrote {naive_path}')

    for opt, predictions in predictions_by_opt.items():
        opt_path = pred_dir / f'{output_stem}-{opt}.pkl'

        with open(opt_path, 'wb') as f:
            pickle.dump(predictions, f)

        print(f'Wrote {opt_path}')


if __name__ == '__main__':
    main()

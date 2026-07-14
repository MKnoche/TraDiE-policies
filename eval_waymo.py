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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from utils import load_data, unpack_npy


T_e = 16
K = 6

EVAL_HORIZONS = (5, 9, 15)

VALID_OBJECT_TYPE_SET = {1, 2, 3}

MISS_THRESHOLDS = {
    5: (1.0, 2.0),
    9: (1.8, 3.6),
    15: (3.0, 6.0),
}

MISS_LAT_THRESH = np.asarray(
    [MISS_THRESHOLDS[h][0] for h in EVAL_HORIZONS],
    dtype=np.float64,
)

MISS_LON_THRESH = np.asarray(
    [MISS_THRESHOLDS[h][1] for h in EVAL_HORIZONS],
    dtype=np.float64,
)

TRAJ_TYPE_UNKNOWN = 0


@dataclass
class ScalarBucket:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class APBucket:
    confidences: List[np.ndarray] = field(default_factory=list)
    true_positive: List[np.ndarray] = field(default_factory=list)
    total: int = 0

    def add(
        self,
        confidence: np.ndarray,
        true_positive: np.ndarray,
        total: int = 1,
    ) -> None:
        self.confidences.append(np.asarray(confidence, dtype=np.float64).reshape(-1))
        self.true_positive.append(np.asarray(true_positive, dtype=np.bool_).reshape(-1))
        self.total += int(total)


@dataclass
class GroundTruthAgent:
    agent_type: int
    category: int
    traj_type: int
    cur_valid: bool
    cur_vel: np.ndarray
    pos: np.ndarray
    head: np.ndarray
    dims_lw: np.ndarray
    valid: np.ndarray


@dataclass
class Prediction:
    traj: np.ndarray
    score: np.ndarray
    order: np.ndarray
    best_idx: int


def _load_gt_agent(agent_data: Dict) -> GroundTruthAgent:
    return GroundTruthAgent(
        agent_type=int(unpack_npy(agent_data['agent_type'])),
        category=int(unpack_npy(agent_data['category'])),
        traj_type=int(unpack_npy(agent_data['traj_type'])),
        cur_valid=bool(unpack_npy(agent_data['cur_valid'])),
        cur_vel=unpack_npy(agent_data['cur_vel']).reshape(2),
        pos=unpack_npy(agent_data['eval_pos']).reshape(T_e, 2),
        head=unpack_npy(agent_data['eval_head']).reshape(T_e),
        dims_lw=unpack_npy(agent_data['eval_dims_lw']).reshape(T_e, 2),
        valid=unpack_npy(agent_data['eval_valid']).reshape(T_e),
    )


def _load_prediction(pred_agent_data: Dict) -> Prediction:
    traj = np.asarray(unpack_npy(pred_agent_data['pos']), dtype=np.float64)
    score = np.asarray(unpack_npy(pred_agent_data['pi']), dtype=np.float64).reshape(-1)

    if traj.shape != (K, T_e, 2):
        raise ValueError(
            f'Expected trajectories to have shape, ({K}, {T_e}, 2), got {traj.shape}'
        )

    if score.shape != (K,):
        raise ValueError(f'Expected scores to have {K} elements, got {len(score)}')

    return Prediction(
        traj=traj,
        score=score,
        order=np.argsort(-score),
        best_idx=int(np.argmax(score)),
    )


def _speed_scale_factor(cur_vel: np.ndarray) -> float:
    speed = float(np.hypot(cur_vel[0], cur_vel[1]))

    if speed < 1.4:
        return 0.5

    if speed > 11.0:
        return 1.0

    return 0.5 + (speed - 1.4) / (11.0 - 1.4) * 0.5


def _displacement_matrix(gt: GroundTruthAgent, pred_traj: np.ndarray) -> np.ndarray:
    pred_xy = pred_traj[:, :T_e, :2]
    return np.linalg.norm(pred_xy - gt.pos[None, :, :], axis=-1)


def _min_ade_fde_by_horizon(
    dists: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    min_ade = np.full(len(EVAL_HORIZONS), np.nan, dtype=np.float64)
    min_fde = np.full(len(EVAL_HORIZONS), np.nan, dtype=np.float64)

    for hi, horizon in enumerate(EVAL_HORIZONS):
        valid_prefix = valid[: horizon + 1]

        if valid_prefix.any():
            ade_per_mode = dists[:, : horizon + 1][:, valid_prefix].mean(axis=1)
            min_ade[hi] = float(ade_per_mode.min())

        if valid[horizon]:
            min_fde[hi] = float(dists[:, horizon].min())

    return min_ade, min_fde


def _match_matrix_eval_horizons(
    gt: GroundTruthAgent,
    pred_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(EVAL_HORIZONS, dtype=np.int64)

    pred_xy = pred_traj[:, idx, :2]
    gt_xy = gt.pos[idx]
    heading = gt.head[idx]

    delta = pred_xy - gt_xy[None, :, :]

    c = np.cos(heading)
    s = np.sin(heading)

    longitudinal = delta[..., 0] * c[None, :] + delta[..., 1] * s[None, :]
    lateral = -delta[..., 0] * s[None, :] + delta[..., 1] * c[None, :]

    scale = _speed_scale_factor(gt.cur_vel)

    matches = (np.abs(lateral / scale) <= MISS_LAT_THRESH[None, :]) & (
        np.abs(longitudinal / scale) <= MISS_LON_THRESH[None, :]
    )

    return matches, gt.valid[idx]


def _ap_samples_from_matches(
    order: np.ndarray,
    scores: np.ndarray,
    matches: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered_match = matches[order].astype(np.bool_, copy=False)
    ordered_conf = scores[order].astype(np.float64, copy=False)

    seen_match_before = np.zeros_like(ordered_match, dtype=np.bool_)
    seen_match_before[1:] = np.maximum.accumulate(ordered_match[:-1])

    hard_tp = ordered_match & ~seen_match_before
    hard_conf = ordered_conf

    soft_keep = ~(ordered_match & seen_match_before)
    soft_conf = ordered_conf[soft_keep]
    soft_tp = ordered_match[soft_keep]

    return hard_conf, hard_tp, soft_conf, soft_tp


def _obb_overlap_many(a_boxes: np.ndarray, b_boxes: np.ndarray) -> np.ndarray:
    a_boxes = np.asarray(a_boxes, dtype=np.float64)
    b_boxes = np.asarray(b_boxes, dtype=np.float64)

    a_center = a_boxes[:, None, 0:2]
    b_center = b_boxes[..., 0:2]
    d = b_center - a_center

    a_heading = a_boxes[:, 2]
    b_heading = b_boxes[..., 2]

    ca, sa = np.cos(a_heading), np.sin(a_heading)
    cb, sb = np.cos(b_heading), np.sin(b_heading)

    a_u = np.stack((ca, sa), axis=-1)[:, None, :]
    a_v = np.stack((-sa, ca), axis=-1)[:, None, :]

    b_u = np.stack((cb, sb), axis=-1)
    b_v = np.stack((-sb, cb), axis=-1)

    a_hl = (a_boxes[:, 3] * 0.5)[:, None]
    a_hw = (a_boxes[:, 4] * 0.5)[:, None]
    b_hl = b_boxes[..., 3] * 0.5
    b_hw = b_boxes[..., 4] * 0.5

    positive_dims = (
        (a_boxes[:, 3] > 0.0)[:, None]
        & (a_boxes[:, 4] > 0.0)[:, None]
        & (b_boxes[..., 3] > 0.0)
        & (b_boxes[..., 4] > 0.0)
    )

    def intervals_overlap(axis: np.ndarray) -> np.ndarray:
        dist = np.abs(np.sum(d * axis, axis=-1))

        ra = a_hl * np.abs(np.sum(a_u * axis, axis=-1)) + a_hw * np.abs(
            np.sum(a_v * axis, axis=-1)
        )

        rb = b_hl * np.abs(np.sum(b_u * axis, axis=-1)) + b_hw * np.abs(
            np.sum(b_v * axis, axis=-1)
        )

        return dist < (ra + rb)

    return (
        positive_dims
        & intervals_overlap(a_u)
        & intervals_overlap(a_v)
        & intervals_overlap(b_u)
        & intervals_overlap(b_v)
    )


def _prediction_heading(best_traj: np.ndarray) -> np.ndarray:
    deltas = best_traj[1:, :2] - best_traj[:-1, :2]
    seg_heading = np.arctan2(deltas[:, 1], deltas[:, 0])

    pred_heading = np.empty(best_traj.shape[0], dtype=np.float64)
    pred_heading[0] = seg_heading[0]
    pred_heading[-1] = seg_heading[-1]

    sin_sum = np.sin(seg_heading[1:]) + np.sin(seg_heading[:-1])
    cos_sum = np.cos(seg_heading[1:]) + np.cos(seg_heading[:-1])
    pred_heading[1:-1] = np.arctan2(sin_sum, cos_sum)

    return pred_heading


def _earliest_overlap_step(
    gt_pos: np.ndarray,
    gt_head: np.ndarray,
    gt_dims_lw: np.ndarray,
    gt_valid: np.ndarray,
    gt_cur_valid: np.ndarray,
    agent_index: int,
    best_traj: np.ndarray,
) -> Optional[int]:
    steps = np.arange(T_e, dtype=np.int64)

    other_mask = gt_cur_valid.copy()
    other_mask[agent_index] = False

    if not other_mask.any():
        return None

    pred_heading = _prediction_heading(best_traj)
    own_dims = gt_dims_lw[agent_index, steps]

    pred_boxes = np.column_stack(
        (
            best_traj[steps, 0],
            best_traj[steps, 1],
            pred_heading[steps],
            own_dims[:, 0],
            own_dims[:, 1],
        )
    )

    other_pos = np.swapaxes(gt_pos[other_mask][:, steps, :], 0, 1)
    other_head = gt_head[other_mask][:, steps].T
    other_dims = np.swapaxes(gt_dims_lw[other_mask][:, steps, :], 0, 1)

    other_boxes = np.stack(
        (
            other_pos[..., 0],
            other_pos[..., 1],
            other_head,
            other_dims[..., 0],
            other_dims[..., 1],
        ),
        axis=-1,
    )

    other_valid = gt_valid[other_mask][:, steps].T

    overlaps = _obb_overlap_many(pred_boxes, other_boxes) & other_valid
    hit_steps = np.flatnonzero(overlaps.any(axis=1))

    return int(hit_steps[0]) if hit_steps.size else None


def _average_precision(bucket: APBucket) -> float:
    if bucket.total == 0 or not bucket.confidences:
        return 0.0

    confidence = np.concatenate(bucket.confidences).astype(np.float64, copy=False)
    true_positive = np.concatenate(bucket.true_positive).astype(np.bool_, copy=False)

    if confidence.size == 0:
        return 0.0

    order = np.lexsort((true_positive.astype(np.int8), -confidence))

    tp = true_positive[order].astype(np.float64)
    cum_tp = np.cumsum(tp)

    n = tp.size
    precision = cum_tp / np.arange(1, n + 1, dtype=np.float64)
    recall = cum_tp / float(bucket.total)

    precision_rev = precision[::-1]
    recall_rev = recall[::-1]

    previous_best = np.empty_like(precision_rev)
    previous_best[0] = -np.inf
    previous_best[1:] = np.maximum.accumulate(precision_rev[:-1])

    record_mask = precision_rev > previous_best
    record_precision = precision_rev[record_mask]
    record_recall = recall_rev[record_mask]

    next_recall = np.empty_like(record_recall)
    next_recall[:-1] = record_recall[1:]
    next_recall[-1] = 0.0

    return float(np.sum(record_precision * (record_recall - next_recall)))


def compute_metrics(
    gt_data: Dict[str, Dict[str, Dict]],
    pred_data: Dict[str, Dict[str, Dict]],
) -> Dict[str, float]:
    """Compute Waymo motion metrics.

    The function evaluates predicted trajectories against the compact Waymo ground-truth
    data. Metrics are accumulated per object type, evaluation horizon, and trajectory
    type where applicable, and are then averaged into the final scalar results.

    The computed metrics are soft mAP, mAP, minADE, minFDE, miss rate, and overlap
    rate.

    Args:
        gt_data:
            Ground-truth data loaded from the compact Waymo GT pickle.
        pred_data:
            Prediction data loaded from a prediction pickle.

    Returns:
        A dictionary mapping metric names to scalar evaluation results.
    """
    min_ade_buckets = defaultdict(ScalarBucket)
    min_fde_buckets = defaultdict(ScalarBucket)
    miss_rate_buckets = defaultdict(ScalarBucket)
    overlap_rate_buckets = defaultdict(ScalarBucket)

    map_buckets = defaultdict(APBucket)
    soft_map_buckets = defaultdict(APBucket)

    for scenario_id, agents_gt in tqdm(
        gt_data.items(),
        smoothing=50 / len(gt_data),
        desc='Evaluating',
    ):
        agent_ids = list(agents_gt.keys())

        gt_agents = [_load_gt_agent(agents_gt[agent_id]) for agent_id in agent_ids]
        agents_pred = pred_data.get(scenario_id, {})

        predictions = {
            agent_id: _load_prediction(agents_pred[agent_id])
            for agent_id in agent_ids
            if agent_id in agents_pred
        }

        gt_pos = np.stack([gt.pos for gt in gt_agents], axis=0)
        gt_head = np.stack([gt.head for gt in gt_agents], axis=0)
        gt_dims_lw = np.stack([gt.dims_lw for gt in gt_agents], axis=0)
        gt_valid = np.stack([gt.valid for gt in gt_agents], axis=0)
        gt_cur_valid = np.asarray([gt.cur_valid for gt in gt_agents], dtype=np.bool_)

        for agent_index, agent_id in enumerate(agent_ids):
            gt = gt_agents[agent_index]

            if gt.agent_type not in VALID_OBJECT_TYPE_SET:
                continue

            if gt.category != 1:
                continue

            pack = predictions[agent_id]

            dists = _displacement_matrix(gt, pack.traj)

            min_ade_values, min_fde_values = _min_ade_fde_by_horizon(
                dists,
                gt.valid,
            )

            matches, valid_horizons = _match_matrix_eval_horizons(
                gt,
                pack.traj,
            )

            earliest_overlap = _earliest_overlap_step(
                gt_pos=gt_pos,
                gt_head=gt_head,
                gt_dims_lw=gt_dims_lw,
                gt_valid=gt_valid,
                gt_cur_valid=gt_cur_valid,
                agent_index=agent_index,
                best_traj=pack.traj[pack.best_idx],
            )

            for hi, horizon in enumerate(EVAL_HORIZONS):
                scalar_key = (gt.agent_type, horizon)

                if np.isfinite(min_ade_values[hi]):
                    min_ade_buckets[scalar_key].add(min_ade_values[hi])

                if np.isfinite(min_fde_values[hi]):
                    min_fde_buckets[scalar_key].add(min_fde_values[hi])

                if valid_horizons[hi]:
                    hard_conf, hard_tp, soft_conf, soft_tp = _ap_samples_from_matches(
                        pack.order,
                        pack.score,
                        matches[:, hi],
                    )

                    miss_rate_buckets[scalar_key].add(0.0 if hard_tp.any() else 1.0)

                    if gt.traj_type != TRAJ_TYPE_UNKNOWN:
                        ap_key = (gt.agent_type, horizon, gt.traj_type)

                        map_buckets[ap_key].add(hard_conf, hard_tp)
                        soft_map_buckets[ap_key].add(soft_conf, soft_tp)

                overlap_rate_buckets[scalar_key].add(
                    1.0
                    if earliest_overlap is not None and earliest_overlap <= horizon
                    else 0.0
                )

    def reduce_scalar(buckets: Dict[Tuple[int, int], ScalarBucket]) -> float:
        values = [bucket.mean() for bucket in buckets.values() if bucket.count]
        return float(np.mean(values)) if values else 0.0

    def reduce_ap(buckets: Dict[Tuple[int, int, int], APBucket]) -> float:
        values = [
            _average_precision(bucket)
            for bucket in buckets.values()
            if bucket.confidences
        ]

        return float(np.mean(values)) if values else 0.0

    return {
        'soft_map': reduce_ap(soft_map_buckets),
        'map': reduce_ap(map_buckets),
        'min_ade': reduce_scalar(min_ade_buckets),
        'min_fde': reduce_scalar(min_fde_buckets),
        'miss_rate': reduce_scalar(miss_rate_buckets),
        'overlap_rate': reduce_scalar(overlap_rate_buckets),
    }


def main() -> None:
    """Evaluate a Waymo prediction pickle against compact ground-truth data.

    Command-line arguments:
        --gt_path:
            Path to the compact Waymo ground-truth pickle.
        --pred_path:
            Path to the prediction pickle to evaluate.

    Outputs:
        The computed metrics printed to stdout.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    args = parser.parse_args()

    gt_data = load_data(Path(args.gt_path))
    pred_data = load_data(Path(args.pred_path))

    metrics = compute_metrics(gt_data, pred_data)

    for name, result in metrics.items():
        print(f'{name: <15}{result:.4f}')


if __name__ == '__main__':
    main()

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
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from utils import pack_npy


T_HIST = 11
T = 80
T_FULL = T_HIST + T
CURRENT_IDX = T_HIST - 1

EVAL_TRACK_INDICES = np.arange(
    CURRENT_IDX + 5,
    T_FULL,
    5,
    dtype=np.int64,
)


TRAJ_TYPE_UNKNOWN = 0
TRAJ_TYPE_STATIONARY = 1
TRAJ_TYPE_STRAIGHT = 2
TRAJ_TYPE_STRAIGHT_RIGHT = 3
TRAJ_TYPE_STRAIGHT_LEFT = 4
TRAJ_TYPE_RIGHT_TURN = 5
TRAJ_TYPE_LEFT_U_TURN = 6
TRAJ_TYPE_LEFT_TURN = 7


def _iter_tfrecord_files(raw_dir: Path):
    for path in sorted(raw_dir.iterdir()):
        if path.is_file():
            yield path


def _iter_scenarios_from_tfrecord(path: Path):
    dataset = tf.data.TFRecordDataset(str(path))

    for raw_record in dataset.as_numpy_iterator():
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record)
        yield scenario


def _get_predicted_track_ids(scenario: Any) -> set[int]:
    predicted_track_ids = set()

    for item in scenario.tracks_to_predict:
        track_index = int(item.track_index)

        if 0 <= track_index < len(scenario.tracks):
            predicted_track_ids.add(scenario.tracks[track_index].id)

    return predicted_track_ids


def _normalize_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _rotate(dx: float, dy: float, heading: float) -> tuple[float, float]:
    c, s = float(np.cos(heading)), float(np.sin(heading))
    return dx * c - dy * s, dx * s + dy * c


def _classify_track_type(
    pos_full: np.ndarray,
    head_full: np.ndarray,
    vel_full: np.ndarray,
    valid_full: np.ndarray,
) -> int:
    if not bool(valid_full[CURRENT_IDX]):
        return TRAJ_TYPE_UNKNOWN

    valid_after = np.flatnonzero(valid_full[CURRENT_IDX + 1 :])

    if valid_after.size == 0:
        return TRAJ_TYPE_UNKNOWN

    last_valid_index = int(valid_after[-1] + CURRENT_IDX + 1)

    start_pos = pos_full[CURRENT_IDX]
    end_pos = pos_full[last_valid_index]

    dx = float(end_pos[0] - start_pos[0])
    dy = float(end_pos[1] - start_pos[1])

    start_head = float(head_full[CURRENT_IDX])
    end_head = float(head_full[last_valid_index])

    heading_diff = _normalize_angle(end_head - start_head)
    local_dx, local_dy = _rotate(dx, dy, -start_head)

    start_vel = vel_full[CURRENT_IDX]
    end_vel = vel_full[last_valid_index]

    max_speed = max(
        float(np.hypot(start_vel[0], start_vel[1])),
        float(np.hypot(end_vel[0], end_vel[1])),
    )

    if max_speed < 2.0 and float(np.hypot(dx, dy)) < 3.0:
        return TRAJ_TYPE_STATIONARY

    if abs(heading_diff) < np.pi / 6.0:
        if abs(local_dy) < 2.5:
            return TRAJ_TYPE_STRAIGHT

        return TRAJ_TYPE_STRAIGHT_RIGHT if local_dy < 0.0 else TRAJ_TYPE_STRAIGHT_LEFT

    if local_dy < 0.0:
        return TRAJ_TYPE_RIGHT_TURN

    if local_dx < 0.0:
        return TRAJ_TYPE_LEFT_U_TURN

    return TRAJ_TYPE_LEFT_TURN


def _summarize_track(
    track: Any, predicted_track_ids: set[int]
) -> dict[str, bytes] | None:
    valid_full = np.zeros((T_FULL,), dtype=np.bool_)
    pos_full = np.zeros((T_FULL, 2), dtype=np.float32)
    head_full = np.zeros((T_FULL,), dtype=np.float32)
    vel_full = np.zeros((T_FULL, 2), dtype=np.float32)
    dims_lw_full = np.zeros((T_FULL, 2), dtype=np.float32)

    num_states = min(len(track.states), T_FULL)

    for t in range(num_states):
        state = track.states[t]

        valid_full[t] = bool(state.valid)

        pos_full[t] = [state.center_x, state.center_y]
        head_full[t] = state.heading
        vel_full[t] = [state.velocity_x, state.velocity_y]
        dims_lw_full[t] = [state.length, state.width]

    if not valid_full[CURRENT_IDX]:
        return None

    agent_type = np.asarray(track.object_type, dtype=np.uint8)

    category = np.asarray(
        1 if track.id in predicted_track_ids else 0,
        dtype=np.uint8,
    )

    cur_vel = vel_full[CURRENT_IDX]
    cur_valid = valid_full[CURRENT_IDX]

    eval_pos = pos_full[EVAL_TRACK_INDICES]
    eval_head = head_full[EVAL_TRACK_INDICES]
    eval_valid = valid_full[EVAL_TRACK_INDICES]

    dims_lw_eval = dims_lw_full[EVAL_TRACK_INDICES]

    traj_type = np.asarray(
        _classify_track_type(
            pos_full=pos_full,
            head_full=head_full,
            vel_full=vel_full,
            valid_full=valid_full,
        )
    )

    return {
        'agent_type': pack_npy(agent_type),
        'category': pack_npy(category),
        'traj_type': pack_npy(traj_type),
        'cur_vel': pack_npy(cur_vel),
        'cur_valid': pack_npy(cur_valid),
        'eval_pos': pack_npy(eval_pos),
        'eval_head': pack_npy(eval_head),
        'eval_valid': pack_npy(eval_valid),
        'eval_dims_lw': pack_npy(dims_lw_eval),
    }


def _summarize_scenario(scenario: Any) -> tuple[str, dict[str, dict[str, bytes]]]:
    scenario_id = str(scenario.scenario_id)
    predicted_track_ids = _get_predicted_track_ids(scenario)

    scene_dict = {}

    for track in scenario.tracks:
        entry = _summarize_track(track, predicted_track_ids)

        if entry is None:
            continue

        agent_id = str(track.id)
        scene_dict[agent_id] = entry

    return scenario_id, scene_dict


def main() -> None:
    """Convert raw Waymo TFRecord scenarios into a compact ground-truth pickle.

    The resulting pickle contains a nested dictionary indexed by scenario ID and agent
    ID. For each evaluated agent, the stored fields summarize the information needed for
    applying TraDiE-policies and evaluation.

    Command-line arguments:
        --raw_dir:
            Path to the directory containing raw Waymo Motion Scenario TFRecord files.
        --gt_path:
            Path where the compact ground-truth pickle should be written.

    Outputs:
        A pickle file at ``--gt_path`` containing a dictionary with the layout
        ``scenario_id -> agent_id -> ground-truth fields``.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.gt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tfrecord_files = list(_iter_tfrecord_files(raw_dir))

    if len(tfrecord_files) == 0:
        raise FileNotFoundError(f'No files found in {raw_dir}')

    data = {}
    n_agents = 0

    for tfrecord_path in tqdm(
        tfrecord_files,
        smoothing=50 / len(tfrecord_files),
        desc='Processing TFRecord files',
    ):
        for scenario in _iter_scenarios_from_tfrecord(tfrecord_path):
            scenario_id, scene_dict = _summarize_scenario(scenario)
            data[scenario_id] = scene_dict
            n_agents += len(scene_dict)

    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'\nWrote: {out_path}')
    print(f'Scenes: {len(data)}')
    print(f'Agents: {n_agents}')


if __name__ == '__main__':
    main()

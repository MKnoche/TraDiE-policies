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
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np
from tqdm import tqdm
from waymo_open_dataset.protos import motion_submission_pb2

from utils import unpack_npy


T_e = 16
K = 8


def _iter_predictions(
    pred_data: Any,
) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
    for scenario_id, agents in pred_data.items():
        for agent_id, payload in agents.items():
            yield (
                str(scenario_id),
                str(agent_id),
                np.asarray(payload['pos']),
                np.asarray(payload['pi']),
            )


def build_submission(
    pred_data: Any,
    *,
    account_name: str,
    unique_method_name: str,
    num_model_parameters: str,
    authors: str,
    description: str,
    uses_lidar_data: bool,
    uses_camera_data: bool,
    uses_public_model_pretraining: bool,
) -> motion_submission_pb2.MotionChallengeSubmission:
    """Build a Waymo Motion Challenge submission proto from prediction data.

    The prediction data is grouped by scenario and converted into a Waymo
    ``MotionChallengeSubmission`` proto.

    Args:
        pred_data:
            Prediction data loaded from a prediction pickle. The data must contain
            packed ``pos`` and ``pi`` arrays for each target agent for each scenario.
        account_name:
            Account name used for the Waymo submission metadata.
        unique_method_name:
            Unique method name used for the Waymo submission metadata.
        num_model_parameters:
            Number of model parameters reported for the submission.
        authors:
            Comma-separated list of submission authors.
        description:
            Text description of the submitted method.
        uses_lidar_data:
            Whether the submitted method uses lidar data.
        uses_camera_data:
            Whether the submitted method uses camera data.
        uses_public_model_pretraining:
            Whether the submitted method uses public model pretraining.

    Returns:
        A Waymo ``MotionChallengeSubmission`` proto containing all scenario and agent
        predictions.
    """
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.MOTION_PREDICTION
    )
    submission.account_name = account_name
    submission.unique_method_name = unique_method_name
    for a in authors.split(','):
        submission.authors.append(a.strip())
    submission.description = description
    submission.uses_lidar_data = uses_lidar_data
    submission.uses_camera_data = uses_camera_data
    submission.uses_public_model_pretraining = uses_public_model_pretraining
    submission.num_model_parameters = num_model_parameters

    grouped: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
    for scenario_id, agent_id, pos, pi in _iter_predictions(pred_data):
        grouped.setdefault(scenario_id, []).append((agent_id, pos, pi))

    for scenario_id, agent_items in tqdm(
        grouped.items(), desc='Building submission protos'
    ):
        scenario_msg = motion_submission_pb2.ChallengeScenarioPredictions()
        scenario_msg.scenario_id = str(scenario_id)

        prediction_set = motion_submission_pb2.PredictionSet()
        for agent_id, pos, pi in agent_items:
            pos = unpack_npy(pos)
            pi = unpack_npy(pi)

            if pos.ndim == 2:
                pos = pos[np.newaxis, ...]
            if pos.ndim != 3 or pos.shape[2] != 2:
                raise ValueError(
                    f'Invalid pos shape for scenario={scenario_id} agent={agent_id}: got {pos.shape}, expected [{K}, {T_e}, 2].'
                )
            if pi.shape[0] != pos.shape[0]:
                raise ValueError(
                    f'Invalid pi length for scenario={scenario_id} agent={agent_id}: got {pi.shape[0]}, expected {pos.shape[0]}.'
                )
            if pos.shape[0] > K:
                topk = np.argsort(-pi)[:K]
                pos = pos[topk]
                pi = pi[topk]

            single_pred = motion_submission_pb2.SingleObjectPrediction()
            try:
                single_pred.object_id = int(agent_id)
            except ValueError as e:
                raise ValueError(
                    f'agent_id must be int-castable for Waymo submission; got {agent_id!r}'
                ) from e

            for mode_traj, conf in zip(pos, pi):
                mode_traj = np.asarray(mode_traj)

                scored_traj = motion_submission_pb2.ScoredTrajectory()
                traj_msg = motion_submission_pb2.Trajectory()
                traj_msg.center_x.extend([float(x) for x in mode_traj[:, 0]])
                traj_msg.center_y.extend([float(y) for y in mode_traj[:, 1]])
                scored_traj.trajectory.CopyFrom(traj_msg)
                scored_traj.confidence = float(conf)
                single_pred.trajectories.append(scored_traj)

            prediction_set.predictions.append(single_pred)

        scenario_msg.single_predictions.CopyFrom(prediction_set)
        submission.scenario_predictions.append(scenario_msg)

    return submission


def main() -> None:
    """Create a Waymo submission TFRecord and archive from a prediction pickle.

    This script loads a prediction pickle produced by the TraDiE policy pipeline,
    converts it into a Waymo Motion Challenge submission proto, writes the serialized
    proto to a TFRecord file, and packages that file into a ``.tar.gz`` archive suitable
    for submission.

    Command-line arguments:
        --pred_path:
            Path to the prediction pickle.
        --out:
            Output prefix. The script writes ``<out>.tfrecord`` and ``<out>.tar.gz``.
        --account_name:
            Account name used for the Waymo submission metadata.
        --unique_method_name:
            Unique method name used for the Waymo submission metadata.
        --num_model_parameters:
            Number of model parameters reported for the submission.
        --uses_lidar_data:
            Flag indicating that the method uses lidar data.
        --uses_camera_data:
            Flag indicating that the method uses camera data.
        --uses_public_model_pretraining:
            Flag indicating that the method uses public model pretraining.
        --authors:
            List of submission authors.
        --description:
            Text description of the submitted method.

    Outputs:
        A serialized Waymo submission proto at ``<out>.tfrecord`` and a compressed
        submission archive at ``<out>.tar.gz``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--account_name', type=str, required=True)
    parser.add_argument('--unique_method_name', type=str, required=True)
    parser.add_argument('--num_model_parameters', type=str, required=True)
    parser.add_argument('--uses_lidar_data', action='store_true')
    parser.add_argument('--uses_camera_data', action='store_true')
    parser.add_argument('--uses_public_model_pretraining', action='store_true')
    parser.add_argument('--authors', type=str, required=True)
    parser.add_argument('--description', type=str, required=True)

    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    out_prefix = Path(args.out)
    out_tfrecord = out_prefix.with_suffix('.tfrecord')
    out_tar = out_prefix.with_suffix('.tar.gz')

    with open(pred_path, 'rb') as f:
        pred_data = pickle.load(f)

    submission = build_submission(
        pred_data,
        account_name=args.account_name,
        unique_method_name=args.unique_method_name,
        num_model_parameters=args.num_model_parameters,
        authors=args.authors,
        description=args.description,
        uses_lidar_data=args.uses_lidar_data,
        uses_camera_data=args.uses_camera_data,
        uses_public_model_pretraining=args.uses_public_model_pretraining,
    )

    out_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    out_tar.parent.mkdir(parents=True, exist_ok=True)

    with open(out_tfrecord, 'wb') as f:
        f.write(submission.SerializeToString())

    with tarfile.open(out_tar, 'w:gz') as tar:
        tar.add(out_tfrecord, arcname=out_tfrecord.name)

    print(f'Wrote: {out_tfrecord}')
    print(f'Wrote: {out_tar}')


if __name__ == '__main__':
    main()

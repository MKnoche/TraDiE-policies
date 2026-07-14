# Towards Metric-Agnostic Trajectory Forecasting

**ECCV 2026**

[arXiv](https://arxiv.org/abs/2607.01133) | [Project Page](https://vision.rwth-aachen.de/TraDiE-policies) | [BibTeX](#Citation)

[Markus Knoche](https://scholar.google.com/citations?user=Kx4v8IMAAAAJ)<sup>1</sup>, [Daan de Geus](https://daandegeus.com/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?hl=de&user=ZcULDB0AAAAJ)<sup>1</sup>

<sup>1</sup> RWTH Aachen University
<sup>2</sup> Eindhoven University of Technology

> [!NOTE] 
> This repository contains code to apply and evaluate model samples using our **TraDiE-policies**. For our model **DONUT-NLL**, check out [this repository](https://github.com/MKnoche/DONUT-NLL).

## Evaluating with TraDiE policies

Clone repository:

```bash
git clone https://github.com/MKnoche/TraDiE-policies.git
cd TraDiE-policies
```

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

### Dimensions

* `S = 3000`: number of samples from your forecasting method
* `K = 6`: number of modes for evaluation
* `T = 80`: number of full future timesteps for Waymo (8s @ 10Hz)
* `T_h = 3`: number of horizons for Waymo (3s, 5s, 8s)
* `T_e = 16`: number of evaluated timesteps for Waymo (8s @ 2Hz)

### Pipeline

#### Create samples pickle from your forecasting method

The expected layout of the pickle is a nested dict `scenario_id -> agent_id -> field -> packed bytes`, where every array-like field is serialized with `pack_npy` from `utils.py`. This ensures compatibility between numpy versions while keeping disk usage low. The policy runner in `apply_policies.py` expects a field `samples_pos` with shape `[S, T_h, 2]` and `samples_head` with shape `[S, T_h]`. If you have a naive forecast baseline you can additionally store `naive_pos` and `naive_pi`, with shapes `[K, T_h, 2]` and `[K]`, respectively.

#### Create Waymo ground-truth pickle from raw GT TFRecords

Use `prepare_waymo_gt.py` to convert the raw Waymo Motion Scenario files into a compact GT pickle. This file stores the current velocity, valid mask, agent type, trajectory type, and evaluation tracks needed by the downstream tools.
```bash
uv run prepare_waymo_gt.py --raw_dir <waymo_raw_dir> --gt_path <gt.pkl>
```

#### Apply TraDiE-policies

Use `apply_policies.py` with both the samples pickle and the GT pickle. The GT file is only required because the `mAP_rectangles` policy uses each agent's current velocity to choose the rectangle size. The script writes one prediction pickle per policy, again using packed arrays in the same scenario/agent layout, with each predicted agent stored as `pos` with shape `[K, T_e, 2]` and `pi` with shape `[K]`.
```bash
uv run apply_policies.py --gt_path <gt.pkl> --sample_path <samples.pkl> --pred_dir <pred_dir> --dataset waymo
```

#### Evaluate

Use `eval_waymo.py` to score any prediction pickle against the GT pickle.

```bash
uv run eval_waymo.py --gt_path <gt.pkl> --pred_path <pred.pkl>
```

#### Create a Waymo submission

Use `create_waymo_submission.py` to convert a prediction pickle into the Waymo submission TFRecord. A `.tar.gz` archive is automatically created for uploading to the benchmark server.

```bash
uv run create_waymo_submission.py --pred_path <pred.pkl> --out <submission_prefix> --account_name ... --unique_method_name ... --num_model_parameters ... --authors ... --description ...
```

## Citation

If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{knoche2026tradie,
  title     = {{Towards Metric-Agnostic Trajectory Forecasting}},
  author    = {Knoche, Markus and de Geus, Daan and Leibe, Bastian},
  booktitle = {ECCV},
  year      = {2026}
}
```
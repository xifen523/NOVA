# NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving 🌟

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12--2.x-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.06254-b31b1b.svg)](https://arxiv.org/abs/2603.06254)

This repository provides the source implementation used for the reported NOVA experiments.

| Release item | Status |
|---|---|
| Source code | Available |
| Pre-trained weights | Coming soon |
| Exact paper-result reproduction | Awaits published weights and required external assets |

[Paper](https://arxiv.org/abs/2603.06254) · Accepted to IROS 2026 · Apache-2.0

Kai Luo¹, Xu Wang², Rui Fan³, and Kailun Yang¹

¹ School of Artificial Intelligence and Robotics, Hunan University

² College of Mechanical and Vehicle Engineering, Hunan University

³ State Key Laboratory of Intelligent Autonomous Systems, Tongji University

NOVA casts online 3D data association as autoregressive next-token prediction.
It serializes recent trajectory context, injects a continuous nine-dimensional
box representation at `<box>` tokens, and converts the language model's
`Yes`/`No` evidence into association costs for Hungarian matching and lifecycle
management.

---

## 🖼️ Teaser

<p align="center">
  <img src="./assets/teaser.jpg" alt="NOVA Teaser" width="100%">
</p>

---

## 🎥 Demo

<video src="https://github.com/user-attachments/assets/2becb52d-cc7b-44a0-83dd-4008a9c99e07" controls="controls" width="100%" height="auto">
  Your browser does not support the video tag.
</video>

---

## Method

NOVA is designed for open-vocabulary 3D multi-object tracking, where category
labels and localization quality can change between Base and Novel objects. Its
association model combines:

- serialized spatio-temporal trajectory context;
- a geometry encoder for `[x, y, z, l, w, h, volume, yaw, score]`;
- hybrid prompting that limits semantic shortcuts for Novel categories;
- hard-negative mining for nearby identity-inconsistent detections;
- an auxiliary IoU/quality head; and
- autoregressive `p_yes`, gating, Hungarian assignment, and track lifecycle.

## Paper-reported results

The following values are transcribed from Table I of
[arXiv:2603.06254v2](https://arxiv.org/abs/2603.06254). They are paper-reported,
not claims that this repository has reproduced every experiment.

| Dataset | Detector | Split | Primary metric | NOVA |
|---|---|---:|---:|---:|
| nuScenes | Find n' Propagate | Base | AMOTA | 48.87 |
| nuScenes | Find n' Propagate | Novel | AMOTA | 22.41 |
| V2X-Seq-SPD | Find n' Propagate + GroundingDINO | Base | sAMOTA | 68.17 |
| V2X-Seq-SPD | Find n' Propagate + GroundingDINO | Novel | sAMOTA | 22.95 |
| KITTI | Find n' Propagate + GroundingDINO | Base | sAMOTA | 93.06 |
| KITTI | Find n' Propagate + GroundingDINO | Novel | sAMOTA | 12.79 |

The preserved Open3DTrack comparison has a known local/paper metric conflict
and is deliberately not presented as a reproduced acceptance result.

## Installation

NOVA requires Python 3.10.

```bash
git clone https://github.com/xifen523/NOVA.git
cd NOVA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[model]"
```

For tests and packaging tools:

```bash
pip install -e ".[model,dev]"
pytest tests/unit tests/integration
```

## Environment

The tested Python 3.10 environment is recorded in
[`environment/environment.yml`](environment/environment.yml), with a tested
dependency snapshot in
[`environment/requirements-tested.txt`](environment/requirements-tested.txt)
and setup notes in [`environment/README.md`](environment/README.md).

## Synthetic quick start

This offline workflow needs no model weights or dataset. The tiny training and
heuristic inference backends are explicit smoke-test choices; real execution is
the CLI default.

```bash
mkdir -p demo-output

nova-prepare \
  --config configs/examples/synthetic_v2x.yaml \
  --input tests/fixtures/synthetic_v2x.json \
  --output demo-output/prepared.jsonl

nova-train \
  --config configs/examples/synthetic_v2x.yaml \
  --input demo-output/prepared.jsonl \
  --output-dir demo-output/train \
  --backend tiny --steps 2

nova-infer \
  --input demo-output/prepared.jsonl \
  --output demo-output/predictions.jsonl \
  --backend heuristic

nova-summarize \
  --input demo-output/predictions.jsonl \
  --output demo-output/summary.json \
  --dataset v2x
```

`nova-summarize` reports structural diagnostics only. Benchmark metrics require
a caller-supplied official or compatible evaluator:

```bash
nova-evaluate --dataset v2x --input tracks.json --output metrics.json \
  --evaluator-name official-v2x --evaluator-version <version> \
  --evaluator-command python /path/to/evaluator.py
```

Without those evaluator arguments, `nova-evaluate` exits with
`EXTERNAL_EVALUATOR_REQUIRED`; it never substitutes prediction counts for
AMOTA-family metrics.

## Real model use

Real loading is offline by default and always uses `trust_remote_code=False`.
Use `--allow-download` only when you intentionally want Hugging Face downloads.

```bash
nova-infer \
  --config configs/examples/v2x_example.yaml \
  --input /path/to/prepared-sequence.json \
  --output tracks.json \
  --mode tracking \
  --model /path/to/model \
  --checkpoint /path/to/nova-checkpoint \
  --device cuda:0 \
  --vocab-alignment strict
```

Checkpoint vocabulary mismatches fail in `strict` mode. The
`verified-overlap` policy is opt-in and only proceeds when tokenizer IDs and
checkpoint embedding rows satisfy the loader's explicit checks.

## Data

NOVA does not redistribute nuScenes, V2X-Seq-SPD, KITTI, detector outputs, or
model weights. Users obtain those assets under their original licenses and
convert detector records into the documented intermediate format.

| Dataset | Input coordinate frame | Paper Base / Novel categories |
|---|---|---|
| nuScenes | global | Base: car, trailer, pedestrian, bicycle; Novel: truck, bus, motorcycle |
| V2X-Seq-SPD | transformed vehicle-side ego LiDAR | Base: car, van, pedestrian, motorcyclist; Novel: bus, truck, cyclist, tricyclist |
| KITTI | calibrated LiDAR | Base: car, cyclist; Novel: pedestrian |

Input schemas and coordinate conventions are defined by the public configs and
fixture formats. Raw official-dataset conversion is intentionally separate from
the prepared detector-format converter.

## Training, inference, summarization, and evaluation

The command-line tools expose the complete public workflow:

```bash
nova-prepare --help
nova-train --help
nova-infer --help
nova-summarize --help
nova-evaluate --help
```

Use the synthetic example above for an offline smoke test. For a real run,
start from the matching configuration in `configs/reproduction/`, prepare the
external detector records, train or provide a compatible checkpoint, run
tracking inference, summarize structural diagnostics, and invoke the official
dataset evaluator through `nova-evaluate`.

## Pretrained weights

This is a code-only release. Pretrained weights are coming soon. The release
interface and required hashes are defined in [`weights/`](weights/); verified
download links will be added after weight publication. Model checkpoints,
adapters, detector assets, and datasets are not stored in this repository.

## Reproduction workflow

The three evidence-qualified configurations are in `configs/reproduction/`.
They separate implemented and verified behavior from required external assets,
values available only after weight release, and historical settings that remain
unresolved.

```bash
nova-prepare \
  --config configs/reproduction/v2x_gd_nova.yaml \
  --input /path/to/detector-results.json \
  --output prepared-input.json

nova-infer --mode tracking \
  --config configs/reproduction/v2x_gd_nova.yaml \
  --model /path/to/verified/base-model \
  --checkpoint /path/to/verified/nova-weight \
  --checkpoint-sha256 <published-checkpoint-tree-sha256> \
  --vocab-alignment verified-overlap \
  --input prepared-input.json \
  --output predictions.json

nova-evaluate \
  --config configs/reproduction/v2x_gd_nova.yaml \
  --prediction predictions.json \
  --ground-truth /path/to/dataset-root \
  --output evaluation.json \
  --evaluator-name <official-evaluator> \
  --evaluator-version <version> \
  --evaluator-command python /path/to/evaluator.py {prediction}
```

The official datasets, detector outputs, base model, future NOVA weights, and
benchmark evaluators are external assets. Exact paper-result reproduction
therefore awaits the published weights and required external assets. Until a
config contains an approved evaluator command, `nova-evaluate --dry-run`
reports `EXTERNAL_EVALUATOR_REQUIRED` and does not substitute synthetic
statistics.

## Citation

```bibtex
@article{luo2026nova,
  title   = {NOVA: Next-step Open-Vocabulary Autoregression for 3D Multi-Object Tracking in Autonomous Driving},
  author  = {Luo, Kai and Wang, Xu and Fan, Rui and Yang, Kailun},
  journal = {arXiv preprint arXiv:2603.06254},
  year    = {2026}
}
```

## ✉️ Contact

For inquiries or potential collaborations, please open an issue or contact
`xifen527@163.com`.

## 📄 License

First-party NOVA code is licensed under the [Apache 2.0 License](LICENSE).
Models, datasets, detector outputs, evaluation tools, and other third-party
assets retain their own terms.

<div align="center">

# LT-Mem

**Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding**

Yumin Lee · Hyoseok Ju · Giseop Kim<br>
DGIST · IROS 2026

[![Project](https://img.shields.io/badge/Project-Page-2563eb)](https://lt-mem.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2608.19059)
[![Dataset](https://img.shields.io/badge/Dataset-LT--VQA-15803d)](https://drive.google.com/drive/folders/1rrwXxJDqJO9P9-wf_-JENX6FP1v9AThC)

[Installation](#installation) · [Run on Lab data](#run-on-lab-data) · [Try with your own data](#try-with-your-own-data) · [Citation](#citation)

</div>

LT-Mem builds persistent object memories from RGB videos across repeated visits. It preserves object identities and change histories through volatility-aware updates, organizing current states, past events and accumulated statistics into **Live, Delta and Meta Memory** for spatial and temporal question answering.

<p align="center">
  <a href="https://lt-mem.github.io/">
    <img src="https://lt-mem.github.io/figure2.png" alt="LT-Mem framework: multi-session perception, object association, volatility-aware updates, and Live/Delta/Meta memory" width="960">
  </a>
</p>

The default memory-building pipeline uses **SAM3** for segmentation, **BGE** for retrieval, and **local Qwen2.5-3B-Instruct** for reasoning. Natural-language QA uses **Gemini** to answer questions about the saved memory.

## Installation

### 1. Set up the environment

- Linux x86_64 with Python 3.10+ and an NVIDIA GPU (**driver ≥ 580**).
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Tested on an RTX 5070 Ti (16 GB VRAM); memory requirements depend on the input.

Run these commands on the **host**. The build installs CUDA 13 and all runtime dependencies, including ChromaDB and LlamaIndex, in Docker; subsequent commands use the repository root as their working directory.

```bash
git clone https://github.com/yumin1102/LT-Mem.git
cd LT-Mem
python3 scripts/build_image.py
python3 scripts/launch.py check
```

### 2. Download the models

Request access to the [official SAM3 checkpoint](https://huggingface.co/facebook/sam3). Once approved, run:

```bash
python3 scripts/launch.py setup
```

When prompted, log in with a [Hugging Face access token](https://huggingface.co/settings/tokens) from your approved account. An existing `HF_TOKEN` is used automatically.

This downloads SAM3, Qwen and BGE into `models/` and creates `qwen.env`. Run setup once; local inference uses these downloaded models.

## Run on Lab data

### 1. Prepare the aligned data

Download the aligned data from [Google Drive](https://drive.google.com/drive/folders/1rrwXxJDqJO9P9-wf_-JENX6FP1v9AThC):

| Dataset | Archive | Destination |
| --- | --- | --- |
| Lab-S | `aligned_data/Lab-S_aligned.zip` | `data/Lab-S_aligned/` |
| Lab-L | `aligned_data/Lab-L_aligned.zip` | `data/Lab-L_Output/` |

From the repository root, extract your downloaded ZIP into `data/` without renaming its folders. Replace `/path/to/` with your download location:

```bash
python3 -m zipfile -e /path/to/Lab-S_aligned.zip data
python3 -m zipfile -e /path/to/Lab-L_aligned.zip data
```

Each Lab preset uses all frames from `robot_1` through `robot_10`:

```text
data/Lab-S_aligned/
├── robot_1/
│   ├── image/000000.png
│   ├── ply/000000.npy
│   └── pose/000000.txt
├── robot_2/...
├── ...
└── robot_10/...
```

### 2. Build memory across sessions

Run either dataset preset:

```bash
python3 scripts/launch.py run --config configs/lab_s.yaml --output_dir results/lab_s
python3 scripts/launch.py run --config configs/lab_l.yaml --output_dir results/lab_l
```

Each preset tracks ten objects across ten sessions, detects changes, and builds Live/Delta/Meta memory. The terminal shows the current object, session and frame progress.

Outputs include event/state CSVs, object locations and `qa_memory.json`. Use a **new output directory** for each run.

### 3. Query saved memory

Natural-language QA defaults to Gemini. Create a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key), then set the key on the host:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
export GEMINI_MODEL="gemini-3.8-flash"
```

You can set `GEMINI_MODEL` to another [available Gemini model](https://ai.google.dev/gemini-api/docs/models) supported by your API key.

Use the memory path for the dataset you ran:

| Dataset | `--memory` |
| --- | --- |
| Lab-S | `results/lab_s/qa_memory.json` |
| Lab-L | `results/lab_l/qa_memory.json` |

For example, after running Lab-L, ask about object locations, histories or movement frequency:

```bash
python3 scripts/launch.py qa \
  --memory results/lab_l/qa_memory.json \
  --question "Where was the robot dog last seen?"
```

Use `--provider qwen` for local natural-language QA. Structured event QA (`--mode events`) and memory building use Qwen by default.

## Try with your own data

https://github.com/user-attachments/assets/8c40dd39-c60f-4dc0-a480-24336fc9fc20

After completing [Installation](#installation), prepare **one RGB MP4 video per visit**. No depth, camera calibration or pose files are required. [MR.ScaleMaster](https://github.com/team-aprl/MR.ScaleMaster) reconstructs and aligns the sessions automatically before LT-Mem builds memory.

### 1. Prepare your videos

Record the same space on each visit, with overlapping views and clear views of the target objects. Put each video in its own session folder and name it `rgb.mp4`:

```text
data/own_data/
├── robot_1/rgb.mp4   # First visit
└── robot_2/rgb.mp4   # Second visit
```

For more visits, add `robot_3`, `robot_4`, … in chronological order.

### 2. Choose the objects to remember

Create `configs/own_data.yaml` with the following content. Replace `red chair` with your target object and a descriptive text prompt:

```yaml
data_root: ../data/own_data
env: OwnData
sessions:
  - [robot_1, Session1]
  - [robot_2, Session2]
objects:
  - name: red chair
    object_classes: [red chair]
```

Include one session entry per video. Add another object entry for each target; each name represents one physical object, so use prompts that distinguish your targets.

### 3. Build and run

Build the full image once; it installs MR.ScaleMaster and its dependencies automatically:

```bash
python3 scripts/build_image.py --target full
```

Then run from the repository root:

```bash
python3 scripts/launch.py run --input data/own_data --config configs/own_data.yaml \
  --output_dir results/own_data
```

Use a new output directory for each run. Event/state results and `qa_memory.json` are saved under `results/own_data/events/`.

<details>
<summary>Try the provided Lab-L videos</summary>

Download `Lab-L_raw.zip` from [Google Drive](https://drive.google.com/drive/folders/1rrwXxJDqJO9P9-wf_-JENX6FP1v9AThC). After building the full image above, run:

```bash
python3 -m zipfile -e /path/to/Lab-L_raw.zip data
python3 scripts/launch.py run --input data/Lab-L_raw --config configs/lab_l.yaml \
  --output_dir results/lab_l_raw
```

The archive contains only RGB videos under `Lab-L_raw/robot_01/rgb.mp4` through `robot_10/rgb.mp4`. The supplied configuration covers all ten sessions.

</details>

## Parking lot example

Extract `aligned_data/Parkinglot_aligned.zip` from [Google Drive](https://drive.google.com/drive/folders/1U11mqQvAy6hFrfCarzxVUrN1dO93-e4R) into `data/`, keeping the `Parking_Output/` folder:

```bash
python3 scripts/launch.py parking --config configs/parking.yaml --output_dir results/parking
```

Saves estimated vehicle counts per session and `qa_memory.json` under `results/parking/`.

## Citation

```bibtex
@inproceedings{lee2026ltmem,
  title     = {LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding},
  author    = {Lee, Yumin and Ju, Hyoseok and Kim, Giseop},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## License

LT-Mem source code is released under the [BSD-3-Clause license](LICENSE). External code and models retain their own licenses, including Qwen Research, SAM and non-commercial MASt3R terms. See [LICENSE](LICENSE) for third-party terms and notices.

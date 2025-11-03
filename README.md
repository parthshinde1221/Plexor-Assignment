# Plexor Assignment
**Junior AI Engineer / Intern Assessment**

This repository demonstrates end-to-end training and inference of the **YOLOv11-S (small)** model using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.  
It was developed as part of the **Plexor Junior AI Engineer assessment**.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Labeling Process](#data-labeling-process)
- [Dataset Preparation & Split](#dataset-preparation--split)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results & Observations](#results--observations)
- [Repository Structure](#repository-structure)

---

## Overview

The goals of this assignment are to:
1. Label a dataset from provided retail surveillance videos.
2. Train a **YOLOv11-S (small)** object detection model.
3. Run inference on test clips and analyze performance.
4. Document the process with clear setup, training, and results.

The assignment focuses on **data labeling accuracy**, **training setup**, and **clarity of documentation**.

> **Note:** Skip the Data Labeling section if you want to directly train or run inference using the provided dataset structure. The pre-labeled dataset is sufficient for training (`train.py`) or testing (`infer.py`).

---

## Installation

### Prerequisites
- **Windows 10/11**
- **Python 3.9+**
- **pip** (latest)
- **Git** (optional)

### Setup (Windows)

1. **Create a working directory**
   ```bash
   mkdir plexor_project && cd plexor_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv yolo_env
   ```

3. **Activate the environment**
   ```bash
   yolo_env\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install ultralytics opencv-python numpy
   ```

5. **Verify installation**
   ```bash
   yolo --help
   ```

> If the `yolo` command isn’t found, try `python -m ultralytics --help`.

---

## Data Labeling Process

> **Note:** You can skip this section if you only want to train or run inference using the provided dataset structure. The pre-labeled dataset is sufficient for training (`train.py`) or testing (`infer.py`). This section helps understand the labeling workflow.

Two videos were provided by Plexor:

| Video              | Duration   | Purpose                           |
|--------------------|------------|-----------------------------------|
| High-quality video | 5 min 50 s | Training / Validation             |
| Low-quality video  | 30 s       | Qualitative testing (no labels)   |

Labeling was performed in **CVAT** using the following procedure:

1. Selected **every 8th frame**, resulting in **~307 labeled frames**.  
2. Used a single class: **`person`**.  
3. Utilized CVAT’s **Track** tool for bounding box interpolation.  
4. Exported annotations in **Ultralytics YOLO Detection 1.0** format.

> Export only the annotated job to avoid empty labels from unlabeled frames.

### Frame Extraction (FFmpeg)
```bash
winget install ffmpeg
ffmpeg -i train_val.mp4 -vf "select=not(mod(n\,8))" -vsync vfr frame_%06d.jpg
```
Run extraction inside your target images directory (e.g., `plexor_dataset/images/train/`).

---

## Dataset Preparation & Split

Ensure your dataset has matching image/label filenames. A helper script (`split_into_val.py`) creates a **70/30 train/val split**.

**Expected structure:**
```
plexor_dataset/
├── data.yaml
├── train.txt
├── val.txt
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── split_into_val.py
```

Run:
```bash
python plexor_dataset/split_into_val.py --root plexor_dataset --val-ratio 0.3 --move
```

---

## Training the Model

Train **YOLOv11s** with pretrained weights:
```powershell
python train.py
```

All training results are stored by default in `./runs_plexor/y11s_person_colab`.  
Validation results are stored in `./runs_plexor/y11s_person_colab_val`.

**Notes**
- Use `device=cpu` to meet inference constraints; GPU (`device=0`) allowed for training.  
- Default setup: `epochs=50`, `imgsz=640`, `batch=8`.  
- Colab GPU training is acceptable — copy back `best.pt` for CPU inference.

---

## Inference

Run inference on a test video or image folder:
```powershell
python infer.py --source test\1720884888.mov
```

To log output while showing it in console:
```powershell
mkdir outputs -ErrorAction SilentlyContinue
python infer.py --source test\1720884888.mov 2>&1 | Tee-Object -FilePath outputs\inference.log
```

Results are saved in `outputs/`.

---

## Command-Line Arguments

### `train.py` knobs
```python
ap.add_argument("--data", default="plexor_dataset/data.yaml")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "0", "1"])
ap.add_argument("--project", default="runs_plexor")
ap.add_argument("--name", default="y11s_person_colab")
```

**Explanation:**
- `--data`: Path to dataset YAML with class info and splits.  
- `--epochs`: Number of training iterations.  
- `--imgsz`: Image resolution for training.  
- `--batch`: Batch size per iteration.  
- `--device`: Device to train on (`auto`, `cpu`, or GPU id).  
- `--project`: Root directory for saving outputs.  
- `--name`: Subdirectory name under `runs_plexor` for this run.

### `infer.py` knobs
```python
ap.add_argument("--weights", default="runs_plexor/y11s_person_colab/weights/best.pt")
ap.add_argument("--source", required=True, help="Path to a video file or an image folder")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "0", "1"])
ap.add_argument("--outdir", default="outputs")
```

**Explanation:**
- `--weights`: Path to trained model checkpoint.  
- `--source`: Input video or image folder.  
- `--conf`: Confidence threshold for detections.  
- `--device`: Select device (`auto`, `cpu`, `0`, `1`).  
- `--outdir`: Directory to store annotated outputs and logs.

---

## Results & Observations

- **Training set:** ~307 images (sampled every 8th frame).  
- **Validation split:** 30%.  
- **Model:** YOLOv11s single-class (`person`).

### 📊 Results Storage & Artifacts

All training, validation, and inference outputs are organized as follows:

#### **Training Results**
```
./runs_plexor/y11s_person_colab/
```
| File | Description |
|------|--------------|
| `results.csv` | Per-epoch metrics (loss, precision, recall, mAP). |
| `results.png` | Graphical training curves. |
| `labels.jpg` | Dataset class distribution. |
| `train_batch*.jpg` | Training samples visualization. |
| `val_batch*_pred.jpg` | Predicted validation results. |
| `Box*_curve.png` | Precision-Recall and F1 score curves. |
| `confusion_matrix.png` | Confusion matrices. |
| `args.yaml` | Configuration summary. |
| `weights/` | Saved model weights (`best.pt`, `last.pt`). |

#### **Validation Results**
```
./runs_plexor/y11s_person_colab_val/
```
Stores validation metrics, confusion matrices, and per-epoch CSV logs.

#### **Inference Results**
```
./outputs/
```
| File | Description |
|------|--------------|
| `*.jpg` / `*.mp4` | Annotated media from inference. |
| `inference.log` | Optional PowerShell console logs. |

> These folders are auto-generated during runs and can be cleaned between sessions.

---

## Repository Structure

```
.
├── plexor_dataset/
│   ├── data.yaml
│   ├── train.txt
│   ├── val.txt
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── split_into_val.py
├── runs_plexor/       # Training & validation outputs
├── outputs/           # Inference results
├── README.md
└── requirements.txt
```
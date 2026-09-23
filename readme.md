# nnDiffusion

nnDiffusion is an extension of nnU-Net v2.7.0 for medical image modality synthesis, with diffusion-style training and inference workflows.

This project was developed and tested on Linux. Compatibility with Windows is not guaranteed.

## 1. Requirements

- Python 3.10
- PyTorch with CUDA support
- A Linux environment
- Approximately 12 GB of GPU memory for training with the provided configuration

## 2. Installation

### 2.1 Create a Python Environment

We recommend using Anaconda to create an isolated environment:

```bash
conda create -n nnDiffusion python=3.10
conda activate nnDiffusion
```

### 2.2 Install PyTorch

Install the PyTorch version that matches your CUDA version and hardware. Please follow the official PyTorch installation instructions for your system.

### 2.3 Install nnDiffusion

Install this repository in editable mode:

```bash
pip install -e .
```

### 2.4 Configure nnU-Net Paths

Set the three nnU-Net environment variables before preprocessing, training, or inference:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## 3. Data Layout

The repository tracks lightweight task manifests under `nnUNet_raw`, but does not track local `.nii.gz` image files. The actual image files should be placed under `nnUNet_raw/data` on your machine.

### 3.1 Expected Directory Structure

```text
nnDiffusion/
└── nnUNet_raw/
    ├── Dataset001_BraTST12T2/              # task manifest: T1 -> T2
    │   ├── dataset.json
    │   └── dataset_test.json
    ├── Dataset002_BraTST12FLAIR/           # task manifest: T1 -> FLAIR
    │   ├── dataset.json
    │   └── dataset_test.json
    ├── ...
    ├── Dataset016_BraTST1CE/               # task manifest: T1/T2/FLAIR -> T1CE
    │   ├── dataset.json
    │   └── dataset_test.json
    └── data/                               # local image files, not tracked by git
        ├── imagesTr/
        │   ├── BraTS-GLI-00000-000/        # one training case folder
        │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz   # T1
        │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz   # T1CE
        │   │   ├── BraTS-GLI-00000-000-t2w.nii.gz   # T2
        │   │   └── BraTS-GLI-00000-000-t2f.nii.gz   # FLAIR
        │   ├── BraTS-GLI-00002-000/
        │   │   └── ...
        │   └── ...
        └── imagesTs/
            ├── BraTS-GLI-00001-000/        # one test case folder
            │   ├── BraTS-GLI-00001-000-t1n.nii.gz
            │   ├── BraTS-GLI-00001-000-t1c.nii.gz
            │   ├── BraTS-GLI-00001-000-t2w.nii.gz
            │   └── BraTS-GLI-00001-000-t2f.nii.gz
            ├── BraTS-GLI-00001-001/
            │   └── ...
            └── ...
```

### 3.2 Manifest Files

Each `DatasetXXX_*` folder defines one modality synthesis task. For example, `Dataset001_BraTST12T2` uses T1 as the source modality and T2 as the target modality.

- `dataset.json` is used for training and includes both `sources` and `targets`.
- `dataset_test.json` is used for inference and includes `sources` only.
- The JSON files are tracked by git.
- The `.nii.gz` files are not tracked by git and must be provided locally.

The manifest paths are relative to their own `DatasetXXX_*` folder. For example:

```text
nnUNet_raw/Dataset001_BraTST12T2/dataset.json
```

points to local image files under:

```text
nnUNet_raw/data/imagesTr/<case_id>/
```

using relative paths such as:

```text
../data/imagesTr/<case_id>/<case_id>-t1n.nii.gz
```

### 3.3 Training Manifest Example

```json
{
  "channel_names": {
    "0": "T1"
  },
  "labels": {
    "0": "T2"
  },
  "numTraining": 1251,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO",
  "dataset": {
    "BraTS-GLI-00000-000": {
      "sources": [
        "../data/imagesTr/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz"
      ],
      "targets": [
        "../data/imagesTr/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz"
      ]
    }
  }
}
```

### 3.4 Inference Manifest Example

```json
{
  "channel_names": {
    "0": "T1"
  },
  "numTest": 219,
  "file_ending": ".nii.gz",
  "dataset": {
    "BraTS-GLI-00001-000": {
      "sources": [
        "../data/imagesTs/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1n.nii.gz"
      ]
    }
  }
}
```

## 4. Training

### 4.1 Planning and Preprocessing

Run planning and preprocessing for the dataset you want to use:

```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID>
```

Replace `<DATASET_ID>` with a dataset ID such as `001`, `008`, or `016`.

Example:

```bash
nnUNetv2_plan_and_preprocess -d 001
```

### 4.2 Model Training

Train one diffusion variant with:

```bash
nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr <TRAINER_NAME>
```

Available diffusion trainers:

```text
nnUNetTrainerDiffusion_ddpm
nnUNetTrainerDiffusion_flow_matching
nnUNetTrainerDiffusion_diffusion_bridge
```

Example:

```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerDiffusion_ddpm
```

## 5. Inference

Run inference with the diffusion prediction script:

```bash
python nnunetv2/inference/predict_from_raw_data_Diffusion.py \
  -i /path/to/dataset_test.json \
  -o /path/to/output \
  -d <DATASET_ID> \
  -tr <TRAINER_NAME> \
  -f 0 \
  -c 3d_fullres \
  -chk checkpoint_final.pth \
  -n 100
```

The `-d`, `-tr`, `-f`, `-c`, and `-chk` values should match the model you trained.

Example:

```bash
python nnunetv2/inference/predict_from_raw_data_Diffusion.py \
  -i nnUNet_raw/Dataset001_BraTST12T2/dataset_test.json \
  -o nnUNet_raw/Dataset001_BraTST12T2/imagesTs_pred_ddpm \
  -d Dataset001_BraTST12T2 \
  -tr nnUNetTrainerDiffusion_ddpm \
  -f 0 \
  -c 3d_fullres \
  -chk checkpoint_final.pth \
  -n 100
```

## 6. Citation

If you use this code, please cite:

```bibtex
@inproceedings{nndiffusion2026pang,
  title={nnDiffusion: A Standardized 3D Diffusion Framework for Medical Image Synthesis},
  author={Pang, Haowen and Zhu, Pengli and Chen, Shannan and Hong, Xiaoming and Ye, Chuyang},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  year={2026},
  organization={Springer}
}
```

### nnU-Net

This project is based on nnU-Net v2.7.0. Please follow the original nnU-Net license terms and cite nnU-Net when using this code:

```text
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.
```

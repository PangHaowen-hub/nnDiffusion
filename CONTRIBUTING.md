# Contributing to nnDiffusion

Thank you for your interest in improving nnDiffusion.

nnDiffusion is an extension of nnU-Net v2.7.0 for medical image modality synthesis with diffusion-style training and
inference workflows. Contributions should preserve compatibility with the underlying nnU-Net workflow whenever possible,
while keeping the diffusion-specific changes clear and maintainable.

## Scope

Contributions are welcome in the following areas:

- fixes for diffusion training, inference, preprocessing, or evaluation
- improvements to modality-synthesis dataset handling
- documentation updates for installation, data layout, training, inference, or evaluation
- small refactors that make the nnDiffusion-specific logic easier to understand
- reproducible bug reports with enough context to diagnose the issue

Large rewrites, major API changes, or changes that break existing nnU-Net command-line workflows should be discussed
before implementation.

## Data and Artifacts

Do not commit local medical image data or generated experiment artifacts.

The repository may include lightweight dataset manifests such as:

- `nnUNet_raw/Dataset*/dataset.json`
- `nnUNet_raw/Dataset*/dataset_test.json`

The repository should not include:

- `.nii.gz`, `.nii`, or other medical image volumes
- model checkpoints such as `.pth`, `.pt`, `.ckpt`, or `.safetensors`
- generated predictions, logs, caches, or local run outputs
- personal environment files or machine-specific paths

Dataset manifests should use relative paths so that users can place their own local data under the documented directory
layout without changing the JSON files.

## Development Workflow

1. Create a dedicated Python environment.
2. Install the project in editable mode with `pip install -e .`.
3. Make focused changes on a separate branch.
4. Keep changes scoped to the issue being addressed.
5. Run the most relevant command or script to validate the change before opening a pull request.

For code changes, include a short explanation of:

- what changed
- why the change is needed
- how it was tested

For bug fixes, include the command, configuration, or dataset condition that reproduced the issue.

## Style

Follow the surrounding code style. Prefer small, direct changes over broad refactors. When adding diffusion-specific
logic to nnU-Net-derived code paths, keep the distinction clear with concise names and comments where helpful.

Documentation should be written so that a new user can reproduce the expected setup without access to private local
paths.

## Upstream Attribution

This project is derived from nnU-Net v2.7.0. Keep the Apache-2.0 license, upstream attribution, and citation information
intact. Do not remove existing upstream copyright or license headers from source files.

For contributions that are intended for the original nnU-Net project rather than nnDiffusion, please refer to the
upstream repository: <https://github.com/MIC-DKFZ/nnUNet>.

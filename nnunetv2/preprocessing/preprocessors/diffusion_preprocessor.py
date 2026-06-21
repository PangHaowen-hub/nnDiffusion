import numpy as np
from typing import Union

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice


class DiffusionPreprocessor(DefaultPreprocessor):
    """
    Diffusion preprocessor for paired modality synthesis.

    Key design choices:
    - Keep source-only cropping behavior aligned between training and inference.
    - Normalize source and target channels independently to [-1, 1].
    - Append normalized target channels to data only when seg/target is present (training path).
    """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        data = data.astype(np.float32, copy=False)
        target = None

        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and target."
            target = seg.astype(np.float32, copy=False)

        # transpose source (and target) first, same order as default nnU-Net
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if target is not None:
            target = target.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop by SOURCE nonzero only -> train/infer consistency
        properties['shape_before_cropping'] = data.shape[1:]
        nonzero_mask = create_nonzero_mask(data)
        bbox = get_bbox_from_mask(nonzero_mask)
        spatial_slicer = bounding_box_to_slice(bbox)
        slicer = (slice(None),) + spatial_slicer

        data = data[slicer]
        if target is not None:
            target = target[slicer]

        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # source normalization params (always available)
        properties['normalization_params'] = []
        for c in range(data.shape[0]):
            lower = np.percentile(data[c], 0.05)
            upper = np.percentile(data[c], 99.95)
            properties['normalization_params'].append({'lower': float(lower), 'upper': float(upper)})
            data[c] = np.clip(data[c], lower, upper)
            eps = 1e-8
            data[c] = (data[c] - lower) / (upper - lower + eps)
            data[c] = data[c] * 2.0 - 1.0

        # target normalization params (training only)
        properties['target_normalization_params'] = []
        if target is not None:
            for c in range(target.shape[0]):
                lower = np.percentile(target[c], 0.05)
                upper = np.percentile(target[c], 99.95)
                properties['target_normalization_params'].append({'lower': float(lower), 'upper': float(upper)})
                target[c] = np.clip(target[c], lower, upper)
                eps = 1e-8
                target[c] = (target[c] - lower) / (upper - lower + eps)
                target[c] = target[c] * 2.0 - 1.0

            # append target as last channels, preserving original diffusion training convention
            data = np.concatenate([data, target], axis=0)

        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)

        # keep dummy seg for nnU-Net dataloader compatibility in training path
        if target is not None:
            seg_out = np.zeros((1, *new_shape), dtype=np.int8)
            properties['class_locations'] = {'-1': []}
        else:
            seg_out = None

        return data, seg_out, properties

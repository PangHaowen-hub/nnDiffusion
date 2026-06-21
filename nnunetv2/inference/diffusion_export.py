import numpy as np
import torch
import os
import re
from typing import Union, List
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.configuration import default_num_processes


def _safe_filename_component(name: str) -> str:
    name = str(name).strip()
    if len(name) == 0:
        return "synth"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def _ordered_values_from_dict(dct: dict) -> List[str]:
    try:
        keys = sorted(dct.keys(), key=lambda k: int(k))
    except Exception:
        keys = sorted(dct.keys())
    return [str(dct[k]) for k in keys]


def _infer_output_channel_names(dataset_json: dict, num_channels: int) -> List[str]:
    # Preferred: explicit target names
    if isinstance(dataset_json.get('target_channel_names', None), dict):
        names = _ordered_values_from_dict(dataset_json['target_channel_names'])
        if len(names) >= num_channels:
            return names[:num_channels]
    if isinstance(dataset_json.get('target_channel_names', None), (list, tuple)):
        names = [str(i) for i in dataset_json['target_channel_names']]
        if len(names) >= num_channels:
            return names[:num_channels]

    # Fallback: labels are used as target modality names in this diffusion setup
    if isinstance(dataset_json.get('labels', None), dict):
        names = _ordered_values_from_dict(dataset_json['labels'])
        if len(names) >= num_channels:
            return names[:num_channels]

    return [f"synth_{i:04d}" for i in range(num_channels)]

def _write_continuous_image_float32(image: np.ndarray, output_fname: str, properties_dict: dict) -> None:
    """
    Save a continuous-valued image as float32, preserving geometry metadata.
    Supports shape [X, Y, Z] or [C, X, Y, Z].
    """
    assert image.ndim in (3, 4)
    image = image.astype(np.float32, copy=False)

    if 'sitk_stuff' in properties_dict:
        import SimpleITK as sitk
        itk_image = sitk.GetImageFromArray(image)
        sitk_meta = properties_dict['sitk_stuff']
        if 'spacing' in sitk_meta:
            itk_image.SetSpacing(sitk_meta['spacing'])
        if 'origin' in sitk_meta:
            itk_image.SetOrigin(sitk_meta['origin'])
        if 'direction' in sitk_meta:
            itk_image.SetDirection(sitk_meta['direction'])
        sitk.WriteImage(itk_image, output_fname, True)
        return

    if 'nibabel_stuff' in properties_dict:
        import nibabel
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

        # Revert to nibabel axis convention (inverse of read-time transpose)
        if image.ndim == 3:
            image_nib = image.transpose((2, 1, 0))
        else:
            image_nib = image.transpose((3, 2, 1, 0))
        nib_meta = properties_dict['nibabel_stuff']

        if 'reoriented_affine' in nib_meta and 'original_affine' in nib_meta:
            img = nibabel.Nifti1Image(image_nib, affine=nib_meta['reoriented_affine'])
            img_ornt = io_orientation(nib_meta['original_affine'])
            ras_ornt = axcodes2ornt("RAS")
            from_canonical = ornt_transform(ras_ornt, img_ornt)
            img = img.as_reoriented(from_canonical)
        else:
            img = nibabel.Nifti1Image(image_nib, affine=nib_meta['original_affine'])

        nibabel.save(img, output_fname)
        return

    raise RuntimeError('Cannot write float image: missing sitk_stuff/nibabel_stuff metadata in properties_dict.')

def convert_predicted_image_to_original_shape(
    predicted_image: Union[torch.Tensor, np.ndarray],
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    properties_dict: dict,
    num_threads_torch: int = default_num_processes
) -> np.ndarray:
    """
    Resample, un-crop and un-transpose a synthesized continuous image.
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # 1. Resample to original shape (after cropping, before resampling)
    # Using 'resampling_fn_data' for continuous values (trilinear/spline)
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    
    target_shape = properties_dict['shape_after_cropping_and_before_resampling']
    
    # configuration_manager.resampling_fn_data accepts (data, new_shape, current_spacing, target_spacing)
    predicted_image = configuration_manager.resampling_fn_data(
        predicted_image,
        target_shape,
        current_spacing,
        spacing_transposed
    )
    
    if isinstance(predicted_image, torch.Tensor):
        predicted_image = predicted_image.cpu().numpy()

    # 2. Revert cropping
    # Put synthesized image back into original volume size
    final_shape = properties_dict['shape_before_cropping']
    # If the network predicted multiple channels, we handle them. Usually C=1 for synthesis.
    num_channels = predicted_image.shape[0]
    
    reverted_cropping = np.zeros((num_channels, *final_shape), dtype=np.float32) - 1.0
    for c in range(num_channels):
        reverted_cropping[c] = insert_crop_into_image(
            reverted_cropping[c], 
            predicted_image[c], 
            properties_dict['bbox_used_for_cropping']
        )

    # 3. Revert transpose
    # Forward transpose was [0, *[i + 1 for i in plans_manager.transpose_forward]]
    # Backward transpose is inverse of transpose_forward
    reverted_transpose = reverted_cropping.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])

    torch.set_num_threads(old_threads)
    return reverted_transpose

def export_diffusion_prediction(
    predicted_array: Union[np.ndarray, torch.Tensor],
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str],
    output_file_truncated: str,
    denormalize: bool = False
):
    """
    Full export pipeline for diffusion synthesis.
    """
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # Convert to original physical space
    final_image = convert_predicted_image_to_original_shape(
        predicted_array, plans_manager, configuration_manager, properties_dict
    )

    # Denormalize if requested.
    # NOTE: at inference-time (no ground-truth target provided), target normalization params
    # are usually unavailable, so denormalization may be skipped safely.
    if denormalize:
        target_params = properties_dict.get('target_normalization_params', None)
        if target_params is not None and len(target_params) >= final_image.shape[0]:
            c = final_image.shape[0]
            denorm = np.empty_like(final_image, dtype=np.float32)
            for i in range(c):
                lower = target_params[i]['lower']
                upper = target_params[i]['upper']
                denorm[i] = ((final_image[i] + 1.0) / 2.0) * (upper - lower) + lower
            final_image = denorm

    # Save as one subject folder containing one or more synthesized images
    file_ending = dataset_json_dict_or_file['file_ending']
    subject_dir = output_file_truncated
    os.makedirs(subject_dir, exist_ok=True)

    if final_image.ndim == 3:
        final_image = final_image[None, ...]

    c = final_image.shape[0]
    names = _infer_output_channel_names(dataset_json_dict_or_file, c)
    names = [_safe_filename_component(n) for n in names]
    seen = {}
    unique_names = []
    for n in names:
        cnt = seen.get(n, 0)
        seen[n] = cnt + 1
        unique_names.append(n if cnt == 0 else f"{n}_{cnt:02d}")
    names = unique_names

    for i in range(c):
        out_name = f"{names[i]}{file_ending}"
        out_path = os.path.join(subject_dir, out_name)
        _write_continuous_image_float32(final_image[i], out_path, properties_dict)

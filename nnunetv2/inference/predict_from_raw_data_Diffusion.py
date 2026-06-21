import os
import torch
import numpy as np
from typing import Union, Tuple
from tqdm import tqdm
from queue import Queue
from threading import Thread
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.diffusion_export import export_diffusion_prediction

class nnUNetPredictorDiffusion(nnUNetPredictor):
    def __init__(self,
                 num_inference_steps: int = 50,
                 eta: float = 0.0,
                 use_ema: bool = True,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, False, perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.use_ema = use_ema
        self.diffusion_process = None
        self.time_encoder = None
        self.target_channels = 1 # Default, will be updated during initialization

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True):
        """
        Diffusion-specific sliding-window accumulation.
        Keep accumulation in float32 to avoid quantization drift across iterative denoising steps.
        """
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            output_channels = int(getattr(self, 'sliding_window_output_channels', self.target_channels))
            predicted_logits = torch.zeros((output_channels, *data.shape[1:]),
                                           dtype=torch.float32,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.float32, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device).to(torch.float32)
            else:
                gaussian = 1.0

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=True) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device, dtype=torch.float32)
                    # Robustly keep diffusion head channels only.
                    if prediction.shape[0] != output_channels:
                        prediction = prediction[:output_channels]

                    if self.use_gaussian:
                        prediction = prediction * gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array during diffusion sliding-window accumulation.')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists,
                                                            seg_from_prev_stage_files,
                                                            output_filenames_truncated,
                                                            num_processes: int):
        """
        Windows-friendly sequential preprocessing iterator.
        Avoids multiprocessing.Manager/Pipe failures (WinError 5) in restricted environments.
        """
        from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot

        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(input_list_of_lists)
        if output_filenames_truncated is None:
            output_filenames_truncated = [None] * len(input_list_of_lists)

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)

        def _iter():
            for files, seg_prev_stage, ofile in zip(input_list_of_lists, seg_from_prev_stage_files, output_filenames_truncated):
                data, seg, data_properties = preprocessor.run_case(
                    files, seg_prev_stage, self.plans_manager, self.configuration_manager, self.dataset_json
                )
                if seg_prev_stage is not None:
                    seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                    data = np.vstack((data, seg_onehot))
                data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
                yield {'data': data, 'data_properties': data_properties, 'ofile': ofile}

        return _iter()

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
                                       folder_with_segs_from_prev_stage: str = None, overwrite: bool = True,
                                       part_id: int = 0, num_parts: int = 1, save_probabilities: bool = False):
        """
        Diffusion-friendly input management:
        - supports standard nnU-Net split naming: case_0000.nii.gz
        - also supports plain single-file inputs: case.nii.gz
        """
        from batchgenerators.utilities.file_and_folder_operations import join, isfile, subfiles

        if isinstance(list_of_lists_or_source_folder, str):
            src = list_of_lists_or_source_folder
            suffix = self.dataset_json['file_ending']
            files = subfiles(src, suffix=suffix, join=True, sort=True)

            # Build case lists in-process (no multiprocessing.Pool), which is
            # robust on restrictive Windows environments.
            split_cases = {}
            plain_cases = []
            for f in files:
                b = os.path.basename(f)
                if b.endswith(suffix):
                    stem = b[:-len(suffix)]
                else:
                    stem = os.path.splitext(b)[0]

                if len(stem) >= 5 and stem[-5] == '_' and stem[-4:].isdigit():
                    case_id = stem[:-5]
                    split_cases.setdefault(case_id, []).append(f)
                else:
                    plain_cases.append([f])

            if len(split_cases) > 0:
                list_of_lists_or_source_folder = [sorted(v) for _, v in sorted(split_cases.items(), key=lambda x: x[0])]
                if len(plain_cases) > 0:
                    list_of_lists_or_source_folder.extend(plain_cases)
            else:
                list_of_lists_or_source_folder = plain_cases
                if len(list_of_lists_or_source_folder) > 0:
                    print("Detected non-splitted input filenames; using one-file-per-case fallback.")

        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]

        caseids = []
        for i in list_of_lists_or_source_folder:
            if len(i) == 0:
                continue
            b = os.path.basename(i[0])
            suffix = self.dataset_json['file_ending']
            if b.endswith('_0000' + suffix):
                caseids.append(b[:-(len(suffix) + 5)])
            elif b.endswith(suffix):
                caseids.append(b[:-len(suffix)])
            else:
                caseids.append(os.path.splitext(b)[0])

        print(
            f'I am processing {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]
        else:
            output_filename_truncated = None

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]

        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')

        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_best.pth'):
        """
        Modified to also initialize diffusion sampler and time encoder.
        """
        # Ensure trainer class has correct target_channels BEFORE parent initialization
        # (parent builds network architecture and loads network weights).
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        f0 = int(use_folds[0])
        checkpoint0 = torch.load(join(model_training_output_dir, f'fold_{f0}', checkpoint_name),
                                 map_location=torch.device('cpu'), weights_only=False)
        trainer_name0 = checkpoint0['trainer_name']
        plans_target_channels = int(checkpoint0.get('init_args', {}).get('plans', {}).get('diffusion_target_channels', 1))

        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2
        trainer_class0 = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                     trainer_name0, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class0 is not None:
            setattr(trainer_class0, 'target_channels', plans_target_channels)

        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        
        # Now instantiate the trainer to get access to sampler and time_encoder
        # We need the last fold's checkpoint for this (or just the class)
        f = use_folds[0]
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)
        
        trainer_name = checkpoint['trainer_name']
        
        # We find the trainer class. Since we are in the same environment, we can import it.
        # But TRAINER_REGISTRY is usually defined in predict_diffusion.py. 
        # Better use nnU-Net's recursive_find_python_class.
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        
        if trainer_class is None:
            raise RuntimeError(f"Could not find trainer class {trainer_name}")

        # Instantiate trainer in a "lightweight" way to get diffusion settings
        # We use the init_args from checkpoint
        init_args = checkpoint['init_args']
        init_args['device'] = self.device
        if 'continue_training' not in init_args['plans']:
             init_args['plans']['continue_training'] = False
             
        trainer = trainer_class(**init_args)
        trainer.initialize()
        trainer.load_checkpoint(join(model_training_output_dir, f'fold_{f}', checkpoint_name))
        if self.use_ema and hasattr(trainer, 'apply_ema_weights_for_inference'):
            used_ema = trainer.apply_ema_weights_for_inference()
            if used_ema:
                print("Using EMA weights for inference.")
                # Make EMA effective for the actual predictor network used in sampling.
                net_for_load = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
                ema_state = trainer._get_network_module().state_dict()
                net_for_load.load_state_dict(ema_state, strict=True)
                if isinstance(self.list_of_parameters, list) and len(self.list_of_parameters) > 0:
                    self.list_of_parameters[0] = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
            else:
                print("EMA weights not found in checkpoint; using raw model weights.")
        
        self.diffusion_process = trainer.diffusion_process
        self.time_encoder = trainer.time_encoder
        self.target_channels = trainer.target_channels
        self.diffusion_mode = trainer.diffusion_mode
        
        print(f"Diffusion Initialized: {self.diffusion_mode}")

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use a model-specific diffusion predictor.")

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 1): # Sequential export for synthesis is often safer for VRAM
        """
        Modified to use export_diffusion_prediction instead of segmentation export.
        """
        # We don't use the export_pool for now to keep it simple and avoid serialization issues with large float32 arrays
        for preprocessed in data_iterator:
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            
            properties = preprocessed['data_properties']
            
            # Run the iterative diffusion synthesis
            prediction = self.predict_logits_from_preprocessed_data(data)

            if ofile is not None:
                print('Sending off for resampling and export...')
                export_diffusion_prediction(
                    prediction, properties, self.configuration_manager, self.plans_manager,
                    self.dataset_json, ofile, denormalize=False
                )
            else:
                return prediction # For single array mode
        
        empty_cache(self.device)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='nnU-Net Diffusion Inference')
    parser.add_argument('-i', type=str, required=True, help='Input folder')
    parser.add_argument('-o', type=str, required=True, help='Output folder')
    parser.add_argument('-d', type=str, required=True, help='Dataset name/ID')
    parser.add_argument('-f', type=int, default=0, help='Fold')
    parser.add_argument('-tr', type=str, required=True,
                        choices=['nnUNetTrainerDiffusion_ddpm',
                                 'nnUNetTrainerDiffusion_flow_matching',
                                 'nnUNetTrainerDiffusion_diffusion_bridge'],
                        help='Trainer class')
    parser.add_argument('-c', type=str, default='3d_fullres', help='Configuration')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='Checkpoint name')
    parser.add_argument('-n', type=int, default=100, help='Inference steps')
    parser.add_argument('--eta', type=float, default=None, help='DDIM eta / bridge posterior noise scale')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA weights at inference.')
    
    args = parser.parse_args()
    

    from nnunetv2.paths import nnUNet_results
    model_folder = join(nnUNet_results, args.d, f"{args.tr}__nnUNetPlansDiffusion__{args.c}")

    if args.tr == 'nnUNetTrainerDiffusion_ddpm':
        from nnunetv2.inference.predict_from_raw_data_DDPM import nnUNetPredictorDiffusionDDPM
        predictor_class = nnUNetPredictorDiffusionDDPM
    elif args.tr == 'nnUNetTrainerDiffusion_flow_matching':
        from nnunetv2.inference.predict_from_raw_data_FlowMatching import nnUNetPredictorDiffusionFlowMatching
        predictor_class = nnUNetPredictorDiffusionFlowMatching
    elif args.tr == 'nnUNetTrainerDiffusion_diffusion_bridge':
        from nnunetv2.inference.predict_from_raw_data_DiffusionBridge import nnUNetPredictorDiffusionBridge
        predictor_class = nnUNetPredictorDiffusionBridge
    else:
        raise RuntimeError(f"Unsupported diffusion trainer: {args.tr}")
    
    eta = args.eta
    if eta is None:
        eta = 1.0 if args.tr == 'nnUNetTrainerDiffusion_diffusion_bridge' else 0.0

    predictor = predictor_class(
        num_inference_steps=args.n,
        eta=eta,
        use_ema=not args.no_ema,
        device=torch.device('cuda'),
        verbose=False
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(args.f,), checkpoint_name=args.chk)

    if os.path.isfile(args.i) and args.i.lower().endswith('.json'):
        test_spec = load_json(args.i)
        if 'dataset' not in test_spec or not isinstance(test_spec['dataset'], dict):
            raise RuntimeError(f"Input json must contain a dict key 'dataset'. Got: {args.i}")

        base_dir = os.path.dirname(os.path.abspath(args.i))
        case_keys = sorted(test_spec['dataset'].keys())
        list_of_lists = []
        output_truncated = []

        for k in case_keys:
            entry = test_spec['dataset'][k]
            if 'sources' not in entry:
                raise RuntimeError(f"Case {k} in {args.i} is missing 'sources'")
            srcs = entry['sources']
            if isinstance(srcs, str):
                srcs = [srcs]
            resolved = [
                s if os.path.isabs(s) else os.path.abspath(os.path.join(base_dir, s))
                for s in srcs
            ]
            list_of_lists.append(resolved)
            output_truncated.append(join(args.o, k))

        predictor.predict_from_files(
            list_of_lists,
            output_truncated,
            save_probabilities=False,
            overwrite=True
        )
    else:
        predictor.predict_from_files(args.i, args.o, save_probabilities=False, overwrite=True)

if __name__ == '__main__':
    main()

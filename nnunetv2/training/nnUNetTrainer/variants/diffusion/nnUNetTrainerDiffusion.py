import math
import torch
from torch import nn
from typing import Union, Tuple, List
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from tqdm import trange
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from einops import rearrange


class TimeConditionedUNetWrapper(nn.Module):
    """
    Wraps the dynamically generated nnU-Net architecture (e.g., PlainConvUNet, ResidualEncoderUNet)
    and intercepts the layers to inject a timestep embedding (t_emb) via 
    Feature-wise Linear Modulation (FiLM) / Adaptive Group Normalization (AdaGN).
    """
    def __init__(self, base_unet, time_emb_dim, num_input_channels, dummy_input_shape=(1, 16, 16, 16)):
        super().__init__()
        self.unet = base_unet
        self.time_emb_dim = time_emb_dim
        
        # Deep supervision flag natively from nnU-Net
        self.predict_3d = len(dummy_input_shape) == 3 # [X, Y, Z]
        
        self._init_spatially_adaptive_projections(num_input_channels, dummy_input_shape)

    @property
    def encoder(self):
        return self.unet.encoder

    @property
    def decoder(self):
        return self.unet.decoder

    def _init_spatially_adaptive_projections(self, num_input_channels, dummy_shape):
        """
        Dynamically passes a dummy tensor to trace the output channels of each encoder 
        and decoder stage, and builds appropriate scale/shift MLPs.
        """
        device = next(self.unet.parameters()).device
        dummy_input = torch.zeros((1, num_input_channels, *dummy_shape), device=device)
        
        self.enc_mlps = nn.ModuleList()
        self.dec_mlps = nn.ModuleList()
        
        # Trace Encoder
        skips = []
        x = dummy_input
        for s in self.unet.encoder.stages:
            x = s(x)
            skips.append(x.clone())
            channels = x.shape[1]
            self.enc_mlps.append(self._build_mlp(channels))
            
        # Trace Decoder
        lres_input = skips[-1]
        for idx, (transpconv, stage) in enumerate(zip(self.unet.decoder.transpconvs, self.unet.decoder.stages)):
            x = transpconv(lres_input)
            x = torch.cat((x, skips[-(idx+2)]), 1)
            x = stage(x)
            lres_input = x
            
            channels = x.shape[1]
            self.dec_mlps.append(self._build_mlp(channels))

    def _build_mlp(self, channels):
        return nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, channels * 2)
        )

    def _apply_film(self, x, t_emb, mlp):
        """
        Applies FiLM/AdaGN to a feature map x.
        t_emb: [B, time_emb_dim]
        mlp: nn.Sequential
        x: [B, C, D, H, W] or [B, C, H, W]
        """
        if t_emb is None:
            return x

        scale_shift = mlp(t_emb) # -> [B, C * 2]
        scale, shift = scale_shift.chunk(2, dim=1) # -> [B, C], [B, C]
        
        # Expand spatial dimensions
        if self.predict_3d:
            scale = rearrange(scale, 'b c -> b c 1 1 1')
            shift = rearrange(shift, 'b c -> b c 1 1 1')
        else:
            scale = rearrange(scale, 'b c -> b c 1 1')
            shift = rearrange(shift, 'b c -> b c 1 1')
            
        return x * (scale + 1) + shift

    def forward(self, x, t_emb=None):
        # --- ENCODER PASS ---
        skips = []
        feat = x
        for stage, mlp in zip(self.unet.encoder.stages, self.enc_mlps):
            feat = stage(feat)
            feat = self._apply_film(feat, t_emb, mlp)
            skips.append(feat)
            
        # --- DECODER PASS ---
        lres_input = skips[-1]
        seg_outputs = []
        
        for idx, (transpconv, stage, mlp) in enumerate(zip(self.unet.decoder.transpconvs, self.unet.decoder.stages, self.dec_mlps)):
            # Upsample
            df = transpconv(lres_input)
            # Concatenate skip connection
            df = torch.cat((df, skips[-(idx+2)]), dim=1)
            # Proceed layer
            df = stage(df)
            
            # Apply FiLM mapping
            df = self._apply_film(df, t_emb, mlp)
            
            # Segmentation Output Head
            if self.unet.decoder.deep_supervision:
                seg_outputs.append(self.unet.decoder.seg_layers[idx](df))
            elif idx == (len(self.unet.decoder.stages) - 1):
                seg_outputs.append(self.unet.decoder.seg_layers[-1](df))
                
            lres_input = df
            
        seg_outputs = seg_outputs[::-1]

        if not self.unet.decoder.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class nnUNetTrainerDiffusion(nnUNetTrainer):
    # ── Class-level defaults ──────────────────────────────────────────────────
    # Subclasses (for DDPM, Flow Matching, Diffusion Bridge) override these to
    # encode the configuration directly in the trainer name, exactly like
    # nnU-Net's built-in trainer variants (e.g. nnUNetTrainer_DP).
    # Priority: class attribute > environment variable > plans dict (checkpoint)
    DIFFUSION_MODE = None
    
    # Diffusion specific hyperparameters as class variables for static access
    num_timesteps = 1000
    time_emb_dim = 64
    target_channels = None
    base_learning_rate = 1e-4
    adamw_weight_decay = 1e-4
    adamw_betas = (0.9, 0.99)
    min_learning_rate = 1e-6
    warmup_epochs = 20
    grad_clip_norm = 1.0
    default_num_epochs = 1000
    default_num_iterations_per_epoch = 500
    default_num_val_iterations_per_epoch = 20
    default_save_every = 25
    ema_enabled_default = True
    ema_decay_default = 0.999
    ema_update_after_step_default = 100
    ema_update_every_default = 1

    @staticmethod
    def _infer_target_channels_from_dataset_json(dataset_json: dict) -> int:
        """
        Infer diffusion target channels from dataset.json without manual CLI args.
        Supported schemas:
        1) diffusion extension: dataset[*].targets (list/str)
        2) legacy/custom:       dataset[*].label   (list/str)
        3) fallback metadata:   labels / target_channel_names
        """
        if not isinstance(dataset_json, dict):
            return 1

        ds = dataset_json.get('dataset', None)
        if isinstance(ds, dict) and len(ds) > 0:
            first_key = next(iter(ds.keys()))
            entry = ds[first_key]
            if isinstance(entry, dict):
                if 'targets' in entry:
                    tgts = entry['targets']
                    return len(tgts) if isinstance(tgts, (list, tuple)) else 1
                if 'label' in entry:
                    lbl = entry['label']
                    return len(lbl) if isinstance(lbl, (list, tuple)) else 1

        if 'target_channel_names' in dataset_json:
            tcn = dataset_json['target_channel_names']
            if isinstance(tcn, dict):
                return len(tcn.keys())
            if isinstance(tcn, (list, tuple)):
                return len(tcn)

        # In your diffusion dataset.json, labels represent target modalities.
        if isinstance(dataset_json.get('labels', None), dict):
            return max(1, len(dataset_json['labels']))

        return 1

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        inferred_tgt = self._infer_target_channels_from_dataset_json(dataset_json)
        
        # Monkey-patch the dataset_json to prevent nnU-Net's strict LabelManager from crashing.
        # Since we use continuous target MRIs instead of categorical labels, user-provided
        # labels (like {"0": "T2"}) will crash LabelManager. We overwrite it with a dummy set.
        if 'labels' in dataset_json:
            dataset_json['labels'] = {'background': 0, 'diffusion_dummy': 1}

        cls_tgt = getattr(self.__class__, 'target_channels', None)
        plan_tgt = plans.get('diffusion_target_channels', None) if isinstance(plans, dict) else None

        if cls_tgt is not None:
            resolved_tgt = int(cls_tgt)
        elif plan_tgt is not None:
            resolved_tgt = int(plan_tgt)
        elif inferred_tgt is not None:
            resolved_tgt = int(inferred_tgt)
        else:
            resolved_tgt = 1

        self.target_channels = resolved_tgt
        self.__class__.target_channels = resolved_tgt

        super().__init__(plans, configuration, fold, dataset_json, device)
        

        # Diffusion-oriented defaults for medical modality translation.
        self.initial_lr = float(self.base_learning_rate)
        self.weight_decay = float(self.adamw_weight_decay)
        self.num_epochs = int(self.default_num_epochs)
        self.num_iterations_per_epoch = int(self.default_num_iterations_per_epoch)
        self.num_val_iterations_per_epoch = int(self.default_num_val_iterations_per_epoch)
        self.save_every = int(self.default_save_every)
        self.ema_enabled = bool(self.ema_enabled_default)
        self.ema_decay = float(self.ema_decay_default)
        self.ema_update_after_step = int(self.ema_update_after_step_default)
        self.ema_update_every = int(self.ema_update_every_default)
        self.ema_num_updates = 0
        self.ema_state_dict = None

    def initialize(self):
        self.diffusion_mode = self.__class__.DIFFUSION_MODE
        self.my_init_kwargs['plans']['diffusion_mode'] = self.diffusion_mode
        self.my_init_kwargs['plans']['diffusion_target_channels'] = int(self.target_channels)

        self.initial_lr = float(self.base_learning_rate)
        self.weight_decay = float(self.adamw_weight_decay)

        # Build network/optimizer/loss using resolved diffusion hyperparameters.
        super().initialize()
        
        self.diffusion_process = self._build_diffusion_process()
             
        self.time_encoder = SinusoidalPositionEmbeddings(dim=self.time_emb_dim).to(self.device)
        if self.ema_enabled and self.ema_state_dict is None:
            self._init_ema_from_model()

    def _build_diffusion_process(self):
        raise NotImplementedError("Each concrete diffusion trainer builds its own diffusion process.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=self.adamw_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        warmup_epochs = max(1, min(int(self.warmup_epochs), max(1, self.num_epochs // 10)))
        min_lr = float(self.min_learning_rate)
        base_lr = float(self.initial_lr)
        min_ratio = max(0.0, min(1.0, min_lr / base_lr))

        def lr_lambda(epoch: int) -> float:
            # Compatible with nnUNetTrainer calling scheduler.step(current_epoch)
            # Warmup from 10% -> 100%, then cosine decay to min_lr.
            if epoch < warmup_epochs:
                progress = float(epoch + 1) / float(max(1, warmup_epochs))
                return 0.1 + 0.9 * progress
            if self.num_epochs <= warmup_epochs:
                return 1.0
            t = float(epoch - warmup_epochs) / float(max(1, self.num_epochs - warmup_epochs))
            t = min(max(t, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler

    def _get_network_module(self):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _init_ema_from_model(self):
        with torch.no_grad():
            model_state = self._get_network_module().state_dict()
            self.ema_state_dict = {k: v.detach().clone() for k, v in model_state.items()}
            self.ema_num_updates = 0

    def _update_ema(self):
        if not self.ema_enabled:
            return
        if self.ema_state_dict is None:
            self._init_ema_from_model()
        self.ema_num_updates += 1
        if self.ema_num_updates <= self.ema_update_after_step:
            return
        if (self.ema_num_updates % self.ema_update_every) != 0:
            return

        with torch.no_grad():
            model_state = self._get_network_module().state_dict()
            decay = float(self.ema_decay)
            one_minus_decay = 1.0 - decay
            for k, v in model_state.items():
                ema_v = self.ema_state_dict[k]
                v_detached = v.detach()
                if torch.is_floating_point(v_detached):
                    ema_v.mul_(decay).add_(v_detached, alpha=one_minus_decay)
                else:
                    ema_v.copy_(v_detached)

    def apply_ema_weights_for_inference(self) -> bool:
        if (not self.ema_enabled) or (self.ema_state_dict is None):
            return False
        self._get_network_module().load_state_dict(self.ema_state_dict, strict=True)
        return True

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        # Override the number of input channels for spatial concatenation
        # Original num_input_channels is just the source modalities 'c'
        # New input: x_t (target_channels) + c (num_input_channels)
        # Note: we use class-level constants here for static architecture building
        cls_tgt = getattr(cls, 'target_channels', None)
        if cls_tgt is not None:
            target_channels = int(cls_tgt)
        else:
            target_channels = int(plans_manager.plans.get('diffusion_target_channels', 1))
        time_emb_dim = cls.time_emb_dim
        
        diffusion_input_channels = num_input_channels + target_channels
        
        # output channels is the noise prediction, which matches target_channels
        diffusion_output_channels = target_channels
        
        base_network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            diffusion_input_channels,
            diffusion_output_channels,
            enable_deep_supervision
        )
        
        patch_size = configuration_manager.patch_size
        return TimeConditionedUNetWrapper(base_network, time_emb_dim, diffusion_input_channels, dummy_input_shape=patch_size)

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        Modality Translation Augmentation Policy.
        
        Removed from nnUNetTrainer defaults (ALL UNSUITABLE for paired image synthesis):
          - GaussianNoiseTransform:         changes source intensity but NOT target → contradictory pairs
          - GaussianBlurTransform:          degrades source content, target is unaffected → wrong pairing
          - MultiplicativeBrightnessTransform: changes source amplitude, target unchanged → breaks [-1,1] consistency
          - ContrastTransform:              same issue as brightness
          - SimulateLowResolutionTransform: designed for segmentation artifact robustness, harmful here
          - GammaTransform (both):          non-linear intensity warp applied only to source
        
        Kept (safe because they apply identically to ALL channels including target):
          - SpatialTransform (rotation + scaling): source & target warp simultaneously → spatial coherence
          - MirrorTransform: source & target mirror simultaneously → spatial coherence
        """
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial transform: rotation + scaling only. SAFE for paired synthesis.
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg='constant',
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # Mirror transform. SAFE for paired synthesis.
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        # Required by nnU-Net data pipeline (removes dummy -1 label from seg)
        transforms.append(RemoveLabelTansform(-1, 0))

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def plot_network_architecture(self):
        """
        Print a compact network summary instead of trying to render a PDF graph.

        hiddenlayer 0.3 uses torch.onnx._optimize_trace which was removed in
        modern PyTorch versions. We use torchinfo (pip install torchinfo) as a
        replacement; if that is also unavailable we fall back to a plain str().
        """
        if self._do_i_compile():
            self.print_to_log_file("Skipping network summary: nnUNet_compile is enabled.")
            return

        if self.local_rank != 0:
            return

        try:
            from torchinfo import summary
            from nnunetv2.utilities.helpers import empty_cache
            from batchgenerators.utilities.file_and_folder_operations import join

            total_in_ch = self.num_input_channels + self.target_channels
            dummy_t     = torch.zeros((1,), device=self.device, dtype=torch.long)
            dummy_t_emb = self.time_encoder(dummy_t)            # [1, time_emb_dim]

            summary_str = str(summary(
                self.network,
                input_data=[
                    torch.rand(1, total_in_ch, *self.configuration_manager.patch_size,
                               device=self.device),
                    dummy_t_emb,
                ],
                depth=4, verbose=0,
            ))

            out_txt = join(self.output_folder, "network_architecture.txt")
            with open(out_txt, "w") as f:
                f.write(summary_str)
            self.print_to_log_file(f"Network summary saved to {out_txt}")
            empty_cache(self.device)

        except ImportError:
            # torchinfo not installed – print a plain text representation
            try:
                self.print_to_log_file("torchinfo not found; printing network as text:")
                self.print_to_log_file(str(self.network))
            except Exception:
                pass
        except Exception as e:
            self.print_to_log_file("Unable to summarise network architecture:")
            self.print_to_log_file(e)

    def _build_loss(self):
        raise NotImplementedError("Each concrete diffusion trainer defines its own loss.")

    def train_step(self, batch: dict) -> dict:
        raise NotImplementedError("Use one of the concrete diffusion trainers.")
        
    def validation_step(self, batch: dict) -> dict:
        raise NotImplementedError("Use one of the concrete diffusion trainers.")

    def on_validation_epoch_end(self, val_outputs: list):
        # Override to prevent standard nnU-Net from attempting to calculate Dice scores
        from nnunetv2.utilities.collate_outputs import collate_outputs
        outputs_collated = collate_outputs(val_outputs)
        
        loss_here = np.mean(outputs_collated['val_loss'])
        
        self.logger.log('val_losses', loss_here, self.current_epoch)
        # We record the negative loss as "dice" so the standard checkpoints ('best_ema')
        # work seamlessly (since nnU-Net saves the model when this metric goes UP).
        self.logger.log('mean_fg_dice', -loss_here, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [-loss_here], self.current_epoch)

    def on_epoch_end(self):
        from time import time
        from batchgenerators.utilities.file_and_folder_operations import join
        
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.get_value('train_losses', step=-1), decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.get_value('val_losses', step=-1), decimals=4))
        self.print_to_log_file('Negative MSE', [np.round(i, decimals=4) for i in
                                               self.logger.get_value('dice_per_class_or_region', step=-1)])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by self.logger
        if self._best_ema is None or self.logger.get_value('ema_fg_dice', step=-1) > self._best_ema:
            self._best_ema = self.logger.get_value('ema_fg_dice', step=-1)
            self.print_to_log_file(f"Yayy! New best Diffusion val_loss (neg_ema): {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self._plot_diffusion_progress()

        self.current_epoch += 1

    def _plot_diffusion_progress(self):
        """
        Diffusion-specific training progress plot.

        Panel 1 – Loss curves
            • train_loss  (blue solid)
            • val_loss    (red solid)
            • EMA val_loss (green solid, smoothed)  ← drives checkpoint selection

        Panel 2 – Learning rate schedule
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from batchgenerators.utilities.file_and_folder_operations import join

        log = self.logger.local_logger.my_fantastic_logging
        epoch = self.current_epoch          # 0-indexed; current epoch just finished

        x = list(range(epoch + 1))

        train_losses = log['train_losses'][:epoch + 1]
        val_losses   = log['val_losses'][:epoch + 1]
        # ema_fg_dice stores -val_loss EMA → negate back to get positive loss EMA
        ema_val_loss  = [-v for v in log['ema_fg_dice'][:epoch + 1]]
        lrs           = log['lrs'][:epoch + 1]

        sns.set_theme(style='darkgrid', font_scale=1.4)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # ── Panel 1 : Loss ────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(x, train_losses, color='steelblue',  lw=2, label='train loss (MSE)')
        ax.plot(x, val_losses,   color='tomato',     lw=2, label='val loss (MSE)')
        ax.plot(x, ema_val_loss, color='limegreen',  lw=2, ls='--',
                label='val loss EMA  (checkpoint criterion)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Diffusion Training – Loss Curves')
        ax.legend(loc='upper right')

        # ── Panel 2 : Learning rate ───────────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(x, lrs, color='orchid', lw=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')

        plt.tight_layout()
        fig.savefig(join(self.output_folder, 'progress.png'), dpi=120)
        plt.close()



    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in trange(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in trange(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self._get_network_module()
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'ema_enabled': self.ema_enabled,
                    'ema_decay': self.ema_decay,
                    'ema_update_after_step': self.ema_update_after_step,
                    'ema_update_every': self.ema_update_every,
                    'ema_num_updates': self.ema_num_updates,
                    'ema_network_weights': self.ema_state_dict if self.ema_state_dict is not None else None,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        super().load_checkpoint(filename_or_checkpoint)

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        self.ema_enabled = bool(checkpoint.get('ema_enabled', self.ema_enabled))
        self.ema_decay = float(checkpoint.get('ema_decay', self.ema_decay))
        self.ema_update_after_step = int(checkpoint.get('ema_update_after_step', self.ema_update_after_step))
        self.ema_update_every = int(checkpoint.get('ema_update_every', self.ema_update_every))
        self.ema_num_updates = int(checkpoint.get('ema_num_updates', self.ema_num_updates))

        ckpt_ema = checkpoint.get('ema_network_weights', None)
        if ckpt_ema is not None:
            self.ema_state_dict = {k: v.to(self.device) for k, v in ckpt_ema.items()}
        elif self.ema_enabled:
            self._init_ema_from_model()

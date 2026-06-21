import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.diffusion.nnUNetTrainerDiffusion import nnUNetTrainerDiffusion
from nnunetv2.training.nnUNetTrainer.variants.diffusion.nnUNetTrainerDiffusion import TimeConditionedUNetWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class DiffusionBridgeSampler:
    def __init__(self, num_timesteps=1000, max_noise=0.1):
        self.num_train_timesteps = num_timesteps
        self.num_timesteps = num_timesteps
        self.max_noise = max_noise

    def _expand(self, value: torch.Tensor, x_shape):
        return value.reshape(value.shape[0], *((1,) * (len(x_shape) - 1)))

    def bridge_mean(self, x0: torch.Tensor, source: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        m_t = self._expand(t.float() / float(self.num_train_timesteps), x0.shape)
        return (1.0 - m_t) * x0 + m_t * source

    def bridge_variance(self, t: torch.Tensor, x_shape) -> torch.Tensor:
        m_t = self._expand(t.float() / float(self.num_train_timesteps), x_shape)
        return (self.max_noise ** 2) * m_t * (1.0 - m_t)

    def q_sample(self, x0: torch.Tensor, source: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        mean = self.bridge_mean(x0, source, t)
        var = self.bridge_variance(t, x0.shape)
        return mean + torch.sqrt(var.clamp_min(0.0)) * noise

    def posterior_sample(
        self,
        x_t: torch.Tensor,
        predicted_x0: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """
        Brownian bridge posterior q(x_s | x_t, x0, source), where s = t_next < t.
        t=T is the deterministic source endpoint, so the first reverse step falls
        back to the bridge marginal at s conditioned on the predicted clean target.
        """
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        mean_next = self.bridge_mean(predicted_x0, source, t_next)
        var_next = self.bridge_variance(t_next, x_t.shape)
        var_t = self.bridge_variance(t, x_t.shape)

        m_t = self._expand(t.float() / float(self.num_train_timesteps), x_t.shape)
        m_next = self._expand(t_next.float() / float(self.num_train_timesteps), x_t.shape)
        mean_t = self.bridge_mean(predicted_x0, source, t)

        ratio = m_next / m_t.clamp_min(1e-8)
        posterior_mean = mean_next + ratio * (x_t - mean_t)
        posterior_var = var_next - ratio.square() * var_t

        at_source_endpoint = (t >= self.num_train_timesteps).reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        posterior_mean = torch.where(at_source_endpoint, mean_next, posterior_mean)
        posterior_var = torch.where(at_source_endpoint, var_next, posterior_var)
        posterior_var = posterior_var.clamp_min(0.0)

        if eta <= 0 or bool(torch.all(t_next == 0).item()):
            return posterior_mean
        return posterior_mean + float(eta) * torch.sqrt(posterior_var) * torch.randn_like(x_t)


class nnUNetTrainerDiffusion_diffusion_bridge(nnUNetTrainerDiffusion):
    """Diffusion Bridge trainer for paired source-to-target translation."""
    DIFFUSION_MODE = 'diffusion_bridge'

    def _build_diffusion_process(self):
        return DiffusionBridgeSampler(num_timesteps=self.num_timesteps)

    @staticmethod
    def _repeat_target_to_source_channels(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        source_channels = source.shape[1]
        target_channels = target.shape[1]
        if source_channels == target_channels:
            return target
        if source_channels % target_channels != 0:
            raise RuntimeError(
                f"DiffusionBridge requires source channels ({source_channels}) to equal target channels "
                f"({target_channels}) or be an integer multiple for multi-bridge averaging."
            )
        repeat_factor = source_channels // target_channels
        return target.repeat_interleave(repeat_factor, dim=1)

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        time_emb_dim = cls.time_emb_dim
        diffusion_input_channels = num_input_channels * 2
        diffusion_output_channels = num_input_channels

        base_network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            diffusion_input_channels,
            diffusion_output_channels,
            enable_deep_supervision
        )

        patch_size = configuration_manager.patch_size
        return TimeConditionedUNetWrapper(
            base_network,
            time_emb_dim,
            diffusion_input_channels,
            dummy_input_shape=patch_size,
        )

    def _build_loss(self):
        class DiffusionBridgeLoss(nn.Module):
            def __init__(self, enable_ds=False, weights=None):
                super().__init__()
                self.enable_ds = enable_ds
                self.weights = weights

            def _single_loss(self, predicted_x0, target_x0):
                return nn.functional.mse_loss(predicted_x0, target_x0)

            def forward(self, predicted_x0, target_x0):
                if self.enable_ds and isinstance(predicted_x0, (list, tuple)):
                    loss = 0
                    for i, pred in enumerate(predicted_x0):
                        if pred.shape != target_x0.shape:
                            target_ds = nn.functional.interpolate(
                                target_x0, size=pred.shape[2:], mode='trilinear', align_corners=False
                            )
                        else:
                            target_ds = target_x0
                        loss += self.weights[i] * self._single_loss(pred, target_ds)
                    return loss
                return self._single_loss(predicted_x0, target_x0)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return DiffusionBridgeLoss(enable_ds=True, weights=weights)
        return DiffusionBridgeLoss(enable_ds=False)

    def train_step(self, batch: dict) -> dict:
        data_all = batch['data'].to(self.device).float()
        source = data_all[:, :-self.target_channels]
        target = data_all[:, -self.target_channels:]
        target_bridge = self._repeat_target_to_source_channels(target, source)

        self.optimizer.zero_grad(set_to_none=True)

        b = source.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=self.device).long()
        noise = torch.randn_like(target_bridge)
        x_t = self.diffusion_process.q_sample(target_bridge, source, t, noise)

        t_emb = self.time_encoder(t)
        net_in = torch.cat([x_t, source], dim=1)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            predicted_x0 = self.network(net_in, t_emb)
            loss = self.loss(predicted_x0, target_bridge)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self._update_ema()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self._update_ema()

        return {'loss': loss.detach().cpu().item()}

    def validation_step(self, batch: dict) -> dict:
        data_all = batch['data'].to(self.device).float()
        source = data_all[:, :-self.target_channels]
        target = data_all[:, -self.target_channels:]
        target_bridge = self._repeat_target_to_source_channels(target, source)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            b = source.shape[0]
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=self.device).long()
            noise = torch.randn_like(target_bridge)
            x_t = self.diffusion_process.q_sample(target_bridge, source, t, noise)

            t_emb = self.time_encoder(t)
            net_in = torch.cat([x_t, source], dim=1)
            predicted_x0 = self.network(net_in, t_emb)
            val_loss = self.loss(predicted_x0, target_bridge)

        return {'val_loss': val_loss.detach().cpu().numpy()}

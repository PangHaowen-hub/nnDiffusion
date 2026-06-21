import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.diffusion.nnUNetTrainerDiffusion import nnUNetTrainerDiffusion
from nnunetv2.utilities.helpers import dummy_context


class FlowMatchingSampler:
    def __init__(self, num_timesteps=1000):
        self.num_train_timesteps = num_timesteps


class nnUNetTrainerDiffusion_flow_matching(nnUNetTrainerDiffusion):
    """Rectified Flow / Flow Matching trainer."""
    DIFFUSION_MODE = 'flow_matching'

    def _build_diffusion_process(self):
        return FlowMatchingSampler(num_timesteps=self.num_timesteps)

    def _build_loss(self):
        class FlowMatchingLoss(nn.Module):
            def __init__(self, enable_ds=False, weights=None):
                super().__init__()
                self.enable_ds = enable_ds
                self.weights = weights

            def _single_loss(self, predicted_velocity, target_velocity):
                return nn.functional.mse_loss(predicted_velocity, target_velocity)

            def forward(self, predicted_velocity, target_velocity):
                if self.enable_ds and isinstance(predicted_velocity, (list, tuple)):
                    loss = 0
                    for i, pred in enumerate(predicted_velocity):
                        if pred.shape != target_velocity.shape:
                            target_ds = nn.functional.interpolate(
                                target_velocity, size=pred.shape[2:], mode='trilinear', align_corners=False
                            )
                        else:
                            target_ds = target_velocity
                        loss += self.weights[i] * self._single_loss(pred, target_ds)
                    return loss
                return self._single_loss(predicted_velocity, target_velocity)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return FlowMatchingLoss(enable_ds=True, weights=weights)
        return FlowMatchingLoss(enable_ds=False)

    def train_step(self, batch: dict) -> dict:
        data_all = batch['data'].to(self.device).float()
        source = data_all[:, :-self.target_channels]
        target = data_all[:, -self.target_channels:]

        self.optimizer.zero_grad(set_to_none=True)

        b = source.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(target)
        tau = (t / self.num_timesteps).float()
        while len(tau.shape) < len(target.shape):
            tau = tau.unsqueeze(-1)
        x_t = tau * target + (1 - tau) * noise
        velocity_target = target - noise

        t_emb = self.time_encoder(t)
        net_in = torch.cat([x_t, source], dim=1)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            predicted_velocity = self.network(net_in, t_emb)
            loss = self.loss(predicted_velocity, velocity_target)

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

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            b = source.shape[0]
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
            noise = torch.randn_like(target)
            tau = (t / self.num_timesteps).float()
            while len(tau.shape) < len(target.shape):
                tau = tau.unsqueeze(-1)
            x_t = tau * target + (1 - tau) * noise
            velocity_target = target - noise

            t_emb = self.time_encoder(t)
            net_in = torch.cat([x_t, source], dim=1)
            predicted_velocity = self.network(net_in, t_emb)
            val_loss = self.loss(predicted_velocity, velocity_target)

        return {'val_loss': val_loss.detach().cpu().numpy()}

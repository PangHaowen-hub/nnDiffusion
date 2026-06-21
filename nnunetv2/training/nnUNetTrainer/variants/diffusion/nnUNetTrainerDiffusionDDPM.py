import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.diffusion.nnUNetTrainerDiffusion import nnUNetTrainerDiffusion
from nnunetv2.utilities.helpers import dummy_context


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMSampler:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.betas = sigmoid_beta_schedule(num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        snr = self.alphas_cumprod / (1 - self.alphas_cumprod + 1e-8)
        self.loss_weight = snr.clamp(max=5.0)

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x_t, t, model_output, clip_x_start=True):
        x_start = model_output
        if clip_x_start:
            x_start.clamp_(-1., 1.)
        inferred_noise = self.predict_noise_from_start(x_t, t, x_start)
        return inferred_noise, x_start


class nnUNetTrainerDiffusion_ddpm(nnUNetTrainerDiffusion):
    """DDPM trainer that predicts the clean image x0."""
    DIFFUSION_MODE = 'ddpm'

    def _build_diffusion_process(self):
        return DDPMSampler(num_timesteps=self.num_timesteps, device=self.device)

    def _build_loss(self):
        class DDPMX0Loss(nn.Module):
            def __init__(self, enable_ds=False, weights=None):
                super().__init__()
                self.enable_ds = enable_ds
                self.weights = weights

            def _single_loss(self, predicted_x0, target_x0, snr_weight):
                loss_raw = nn.functional.mse_loss(predicted_x0, target_x0, reduction='none')
                loss_per_sample = loss_raw.flatten(1).mean(dim=1)
                weight_per_sample = snr_weight.flatten(1).mean(dim=1)
                return (loss_per_sample * weight_per_sample).mean()

            def forward(self, predicted_x0, target_x0, snr_weight):
                if self.enable_ds and isinstance(predicted_x0, (list, tuple)):
                    loss = 0
                    for i, pred in enumerate(predicted_x0):
                        if pred.shape != target_x0.shape:
                            target_ds = nn.functional.interpolate(
                                target_x0, size=pred.shape[2:], mode='trilinear', align_corners=False
                            )
                        else:
                            target_ds = target_x0
                        loss += self.weights[i] * self._single_loss(pred, target_ds, snr_weight)
                    return loss
                return self._single_loss(predicted_x0, target_x0, snr_weight)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return DDPMX0Loss(enable_ds=True, weights=weights)
        return DDPMX0Loss(enable_ds=False)

    def train_step(self, batch: dict) -> dict:
        data_all = batch['data'].to(self.device).float()
        source = data_all[:, :-self.target_channels]
        x0 = data_all[:, -self.target_channels:]

        self.optimizer.zero_grad(set_to_none=True)

        b = source.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = (
            extract(self.diffusion_process.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract(self.diffusion_process.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        target_x0 = x0
        snr_weight = extract(self.diffusion_process.loss_weight, t, x0.shape)

        t_emb = self.time_encoder(t)
        net_in = torch.cat([x_t, source], dim=1)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            predicted_x0 = self.network(net_in, t_emb)
            loss = self.loss(predicted_x0, target_x0, snr_weight)

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
        x0 = data_all[:, -self.target_channels:]

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            b = source.shape[0]
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
            noise = torch.randn_like(x0)
            x_t = (
                extract(self.diffusion_process.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.diffusion_process.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
            )
            target_x0 = x0
            snr_weight = extract(self.diffusion_process.loss_weight, t, x0.shape)

            t_emb = self.time_encoder(t)
            net_in = torch.cat([x_t, source], dim=1)
            predicted_x0 = self.network(net_in, t_emb)
            val_loss = self.loss(predicted_x0, target_x0, snr_weight)

        return {'val_loss': val_loss.detach().cpu().numpy()}

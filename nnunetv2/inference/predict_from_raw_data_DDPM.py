import numpy as np
import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data_Diffusion import nnUNetPredictorDiffusion


class nnUNetPredictorDiffusionDDPM(nnUNetPredictorDiffusion):
    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        device = self.device
        shape = (self.target_channels, *data.shape[1:])
        total_T = self.diffusion_process.num_timesteps
        source_cond = data.to(device).float()
        img = torch.randn(shape, device=device)

        timesteps = np.linspace(total_T - 1, 0, self.num_inference_steps, dtype=np.int64)
        timesteps_prev = np.append(timesteps[1:], -1)

        loop = tqdm(range(self.num_inference_steps), desc="DDPM Sampling", disable=not self.allow_tqdm)
        for step_i in loop:
            t_curr = int(timesteps[step_i])
            t_prev = int(timesteps_prev[step_i])
            t = torch.full((1,), t_curr, device=device, dtype=torch.long)
            t_emb = self.time_encoder(t)

            current_input = torch.cat([img, source_cond], dim=0)
            original_forward = self.network.forward
            self.network.forward = lambda x: original_forward(x, t_emb)
            try:
                predicted_x0 = self.predict_sliding_window_return_logits(current_input).to(device)
                if predicted_x0.shape[0] != self.target_channels:
                    predicted_x0 = predicted_x0[:self.target_channels]
            finally:
                self.network.forward = original_forward

            t_tensor = torch.full((1,), t_curr, device=device, dtype=torch.long)
            inferred_noise, x_start = self.diffusion_process.model_predictions(
                img, t_tensor, predicted_x0, clip_x_start=True
            )

            if t_prev < 0:
                img = x_start
                continue

            alpha = self.diffusion_process.alphas_cumprod[t_curr]
            alpha_next = self.diffusion_process.alphas_cumprod[t_prev]

            while alpha.ndim < img.ndim:
                alpha = alpha.unsqueeze(-1)
                alpha_next = alpha_next.unsqueeze(-1)

            sigma = self.eta * torch.sqrt(
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            )
            c = torch.sqrt(1 - alpha_next - sigma ** 2)
            noise = torch.randn_like(img) if self.eta > 0 else 0.
            img = x_start * torch.sqrt(alpha_next) + c * inferred_noise + sigma * noise

        return img.cpu()

import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data_Diffusion import nnUNetPredictorDiffusion


class nnUNetPredictorDiffusionFlowMatching(nnUNetPredictorDiffusion):
    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        device = self.device
        total_T = self.diffusion_process.num_train_timesteps
        source_cond = data.to(device).float()
        shape = (self.target_channels, *data.shape[1:])
        img = torch.randn(shape, device=device)

        timesteps = torch.linspace(0, total_T, self.num_inference_steps + 1).long()
        loop = tqdm(range(self.num_inference_steps), desc="Flow Matching Sampling", disable=not self.allow_tqdm)

        for step_i in loop:
            t_curr = timesteps[step_i].item()
            t_next = timesteps[step_i + 1].item()
            t = torch.full((1,), t_curr, device=device, dtype=torch.long)
            t_emb = self.time_encoder(t)

            current_input = torch.cat([img, source_cond], dim=0)
            original_forward = self.network.forward
            self.network.forward = lambda x: original_forward(x, t_emb)
            try:
                predicted_velocity = self.predict_sliding_window_return_logits(current_input).to(device)
                if predicted_velocity.shape[0] != self.target_channels:
                    predicted_velocity = predicted_velocity[:self.target_channels]
            finally:
                self.network.forward = original_forward

            dt = (t_next - t_curr) / total_T
            img = img + predicted_velocity * dt

        return img.cpu()

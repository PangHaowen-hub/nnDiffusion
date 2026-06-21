import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data_Diffusion import nnUNetPredictorDiffusion


class nnUNetPredictorDiffusionBridge(nnUNetPredictorDiffusion):
    def __init__(self, *args, eta: float = 1.0, **kwargs):
        super().__init__(*args, eta=eta, **kwargs)

    def _average_bridge_channels_to_target(self, img: torch.Tensor) -> torch.Tensor:
        source_channels = img.shape[0]
        if source_channels == self.target_channels:
            return img
        if source_channels % self.target_channels != 0:
            raise RuntimeError(
                f"DiffusionBridge cannot average {source_channels} bridge channels into "
                f"{self.target_channels} target channels."
            )
        repeat_factor = source_channels // self.target_channels
        return img.reshape(self.target_channels, repeat_factor, *img.shape[1:]).mean(dim=1)

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        device = self.device
        total_T = self.diffusion_process.num_train_timesteps
        source_cond = data.to(device).float()
        source_channels = source_cond.shape[0]
        if source_channels % self.target_channels != 0:
            raise RuntimeError(
                f"DiffusionBridge requires source channels ({source_channels}) to equal target channels "
                f"({self.target_channels}) or be an integer multiple for multi-bridge averaging."
            )
        img = source_cond.clone()
        self.sliding_window_output_channels = source_channels

        timesteps = torch.linspace(total_T, 0, self.num_inference_steps + 1).long()
        loop = tqdm(range(self.num_inference_steps), desc="Diffusion Bridge Posterior Sampling", disable=not self.allow_tqdm)

        for step_i in loop:
            t_curr = timesteps[step_i].item()
            t_next = timesteps[step_i + 1].item()
            if t_next == t_curr:
                continue
            t = torch.full((1,), t_curr, device=device, dtype=torch.long)
            t_prev = torch.full((1,), t_next, device=device, dtype=torch.long)
            t_emb = self.time_encoder(t)

            current_input = torch.cat([img, source_cond], dim=0)
            original_forward = self.network.forward
            self.network.forward = lambda x: original_forward(x, t_emb)
            try:
                predicted_x0 = self.predict_sliding_window_return_logits(current_input).to(device)
                if predicted_x0.shape[0] != source_channels:
                    predicted_x0 = predicted_x0[:source_channels]
            finally:
                self.network.forward = original_forward

            img = self.diffusion_process.posterior_sample(
                img.unsqueeze(0),
                predicted_x0.unsqueeze(0),
                source_cond.unsqueeze(0),
                t,
                t_prev,
                eta=self.eta,
            ).squeeze(0)

        return self._average_bridge_channels_to_target(img).cpu()

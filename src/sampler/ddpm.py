import torch
import numpy as np

class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        training_steps = 1000, 
        beta_start: float = 0.00085, 
        beta_end: float = 0.012
        ):

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.training_steps = training_steps
        self.timesteps = torch.from_numpy(np.arange(0, training_steps)[::-1].copy())
        
    def set_steps(self, steps):
        self.steps = steps
        #if steps = 50, then timesteps = 999, 999-20, 999-40, 999-60, ..., 0 = 50 steps

        step_ratio = self.training_steps // steps
        timesteps = np.arange(0, self.training_steps, step_ratio)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.training_steps // self.steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_cumprod_t = self.alpha_cumprod[timestep]
        alpha_cumprod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        current_beta_t = 1 - alpha_cumprod_t / alpha_cumprod_prev_t

        # !!!!!! CHECK THIS FORMULA IN DDPM PAPER !!!!!!!
        # forumla 7 in DDPM paper
        variance = (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=0.7):
        start_step = self.training_steps - int(self.training_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_cumprod_t = 1 - alpha_cumprod_t
        beta_cumprod_prev_t = 1 - alpha_cumprod_prev_t

        current_alpha_t = alpha_cumprod_t / alpha_cumprod_prev_t
        current_beta_t = 1 - current_alpha_t

        # Compute the predicted original sample 
        # !!!!!!!!!!!!!!!!!!!! CHECK DDPM PAPER FOR THIS !!!!!!!!!!!!!!!!!!!!!!! 
        predicted_original_sample = (latents - beta_cumprod_t ** 0.5 * model_output) / alpha_cumprod_t ** 0.5 # x0 (formula 15 in DDPM paper)

        # Compute the coefficient for predicted original sample and current sample x_t
        predicted_original_sample_coeff = (alpha_cumprod_prev_t ** 0.5 * current_beta_t) / beta_cumprod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_cumprod_prev_t / beta_cumprod_t
        
        # Computed the predicted previous sample mean
        predicted_previous_sample = predicted_original_sample * predicted_original_sample_coeff + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator = self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise # N(0, 1) -> N(mu, sigma^2)  || X = mu + sigma * Z, Z ~ N(0, 1)

        return predicted_previous_sample



    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()

        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timesteps]) ** 0.5 # we want standard deviation, not variance so we take square root
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()

        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        # Z = N(0, 1) -> N(mean, variance) = X
        # X = mean + stdev * Z
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_sample.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_samples

    


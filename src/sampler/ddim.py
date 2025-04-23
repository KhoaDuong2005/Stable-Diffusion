import torch
import numpy as np

def unnormalize_to_zero_to_one(t):
    """Converts tensors from (-1,1) to (0,1)."""
    return (t + 1) * 0.5

class DDIMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        training_steps = 1000,
    ):
        # Linear beta schedule
        scale = 1000 / training_steps
        beta_start_scaled = scale * 0.0001
        beta_end_scaled = scale * 0.02
        self.betas = torch.linspace(beta_start_scaled, beta_end_scaled, training_steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        # Store parameters
        self.generator = generator
        self.training_steps = training_steps
        self.timesteps = torch.from_numpy(np.arange(0, training_steps)[::-1].copy())
        
        # Default to 50 sampling steps as in original DDIM paper
        self.steps = 50
        self.set_steps(self.steps)
    
    def set_steps(self, steps):
        self.steps = steps
        step_ratio = self.training_steps // steps
        timesteps = (np.arange(0, steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.training_steps // self.steps)
        return max(prev_t, 0)
    
    def set_strength(self, strength=0.7):
        start_step = self.training_steps - int(self.training_steps * strength)
        self.timesteps = self.timesteps[self.timesteps >= start_step]
        self.start_step = start_step
        
    def _get_variance(self, timestep: int) -> torch.Tensor:
        return torch.tensor(0.0) 
        
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        device = latents.device
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        alpha_cumprod_t = self.alpha_cumprod[t].to(device)
        alpha_cumprod_prev_t = self.alpha_cumprod[prev_t].to(device) if prev_t >= 0 else self.one.to(device)
        
        # Predict x0
        epsilon_t = model_output
        
        # Apply reshaping for broadcasting
        alpha_t = alpha_cumprod_t.reshape(-1, 1, 1, 1) if len(latents.shape) > 2 else alpha_cumprod_t
        alpha_t_next = alpha_cumprod_prev_t.reshape(-1, 1, 1, 1) if len(latents.shape) > 2 else alpha_cumprod_prev_t
        
        # predict x0
        x0_t = (latents - epsilon_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        c2 = (1 - alpha_t_next).sqrt()
        
        xt_next = alpha_t_next.sqrt() * x0_t + c2 * epsilon_t
        
        return xt_next
    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """Add noise to samples at specified timesteps."""
        device = original_samples.device
        alpha_cumprod = self.alpha_cumprod.to(device=device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device)
        
        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)
        
        return noisy_samples
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import numpy as np



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, z_dim=4):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        self.down1 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        )
        
        self.down2 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        )
        
        self.down3 = nn.Sequential(
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        )
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, z_dim * 2, kernel_size=3, padding=1)
        )
        
    def forward(self, x, noise):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        std = torch.exp(0.5 * log_var)
        
        z = mean + std * noise
        z *= 0.18215
        
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim=4):
        super().__init__()
        self.conv_in = nn.Conv2d(z_dim, 256, kernel_size=3, padding=1)
        
        self.bottleneck = ResidualBlock(256, 256)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 256)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64)
        )
        
        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, z):
        z /= 0.18215
        x = self.conv_in(z)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.output(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, z_dim=4):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        
    def forward(self, x, noise_scale=1.0):
        batch_size, _, height, width = x.shape
        device = x.device
        
        noise = torch.randn(batch_size, 4, height // 8, width // 8, device=device) * noise_scale
        
        z = self.encoder(x, noise)
        reconstructed = self.decoder(z)
        
        return reconstructed

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.images = []
        
        file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))
        
        self.images = file_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except:
            return torch.zeros((3, 32, 32))

def inference(model, num_samples=1, device="cpu", image_size=32, latent_dim=4):
    model.eval()
    latent_h = image_size // 8
    latent_w = image_size // 8
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, latent_h, latent_w).to(device)
        z *= 0.18215 
        generated_images = model.decoder(z)
        generated_images = generated_images.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
        generated_images = np.clip(generated_images * 255, 0, 255).astype(np.uint8)
        pil_images = [Image.fromarray(img) for img in generated_images]
        return pil_images
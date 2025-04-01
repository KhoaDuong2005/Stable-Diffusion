import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from encoder import VAE_Encoder
from decoder import VAE_Decoder
import os
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        noise = torch.randn(batch_size, 4, height // 8, width // 8).to(x.device)

        latents = self.encoder(x, noise)

        reconstructed = self.decoder(latents)

        return reconstructed, latents, noise
        
    
    def loss_function(self, recon_x, x, latents, noise):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + noise - noise.pow(2) - noise.exp())

        return recon_loss + 0.0001 * kl_loss, recon_loss, kl_loss


def train_vae(epochs=30, batch_size=32, lr=1e-4, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # TODO: Implement an UI for selecting the dataset
    dataset = datasets.ImageFolder("../data/images", transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)

    os.makedirs("../vae_checkpoints", exist_ok=True)
    
    vae = VAE().to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5) 

    for epoch in tqdm(range(epochs), desc="Training VAE", unit="epoch"):
        vae.train()
        total_loss = 0
        recon_loss = 0
        kl_loss = 0

        for batch in dataloader:
            images, _ = batch
            images = images.to(device)

            optimizer.zero_grad()

            recon_images, latents, noise = vae(images)

            loss, recon_loss, kl_loss = vae.loss_function(recon_images, images, latents, noise)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += recon_loss.item()
            kl_loss += kl_loss.item()


            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = recon_loss / len(dataloader)
        avg_kl_loss = kl_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")

        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save({
                "encoder_state_dict": vae.encoder.state_dict(),
                "decoder_state_dict": vae.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, f"../vae_checkpoints/vae_epoch_{epoch}.pth")

    

train_vae()
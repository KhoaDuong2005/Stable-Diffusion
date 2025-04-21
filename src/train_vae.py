import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from torch.amp import GradScaler, autocast

from vae import VariationalAutoencoder, inference, ImageDataset

import argparse
import os

parser = argparse.ArgumentParser(description="Train a Variational Autoencoder")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing images")
parser.add_argument("--image_size", type=int, default=128, help="Size of the images")
parser.add_argument("--latent_dim", type=int, default=4, help="Dimensionality of the latent space")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")



if __name__ == '__main__':

    args = parser.parse_args()
    
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True


    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = args.image_size
    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR_RATE = 3e-4
    NUM_WORKERS = 8

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # data_dir = r"D:\Project\Stable-Diffusion\data\images"
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        exit(1)
        
    dataset = ImageDataset(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)

    model = VariationalAutoencoder(z_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_RATE, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)
    recon_loss_fn = nn.MSELoss(reduction="sum")
    
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        epoch_recon_loss = 0
        
        for i, x in loop:
            x = x.to(DEVICE, non_blocking=True)
            
            if x.shape[0] == 1:
                continue

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                x_reconstructed = model(x) 
                reconstruction_loss = recon_loss_fn(x_reconstructed, x) / x.shape[0]
            
            scaler.scale(reconstruction_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            epoch_recon_loss += reconstruction_loss.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(ReconLoss=reconstruction_loss.item()) 
        
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Recon Loss: {avg_recon_loss:.4f}") 

        model.eval()
        with torch.no_grad():
            rand_idx = np.random.randint(0, len(dataset))
            real_img = dataset[rand_idx].unsqueeze(0).to(DEVICE)
            
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                recon_img = model(real_img) 

            real_display = real_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
            recon_display = recon_img[0].cpu().permute(1, 2, 0).float().numpy() * 0.5 + 0.5

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(np.clip(real_display, 0, 1))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(np.clip(recon_display, 0, 1))
            axs[1].set_title("Reconstruction")
            axs[1].axis("off")

            plt.tight_layout()
            save_path = f"recon_epoch_{epoch+1}.png"
            plt.savefig(save_path)
            plt.close()

    model_save_path = "lite_vae_model.pt"
    torch.save(model.state_dict(), model_save_path)
    
    generated_images = inference(model, num_samples=4, device=DEVICE, image_size=IMAGE_SIZE, latent_dim=LATENT_DIM)
    
    fig, axes = plt.subplots(1, len(generated_images), figsize=(12, 4))
    if len(generated_images) == 1:
        axes = [axes]
    for ax, img in zip(axes, generated_images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.close()
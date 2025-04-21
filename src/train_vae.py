import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from vae import VariationalAutoEncoder
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 3 * 64 * 64
HIDDEN_DIM = 512
Z_DIM = 64
EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 1e-4

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class NumpyDataset(Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image_np)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label
    
def load_image_data(data_dir: str):
    images = []
    labels = []
    label_names = sorted([directory for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))])
    
    label_index = {}
    for index, label_name in enumerate(label_names):
        label_index[label_name] = index

    for label_name in label_names:
        folder_path = os.path.join(data_dir, label_name)
        for file in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    image = Image.open(file_path).convert("RGB")
                    image_np = np.array(image)
                    images.append(image_np)
                    labels.append(label_index[label_name])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return images, labels, label_index

data_dir = r"D:\Project\Stable-Diffusion\data\images"

images, labels, label_index = load_image_data(data_dir)
dataset = NumpyDataset(images, labels, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(INPUT_DIM, HIDDEN_DIM, Z_DIM).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-5)
recon_loss_fn = nn.MSELoss(reduction="mean")
beta = 0.001

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader)
    epoch_loss = 0
    
    for i, (x, _) in enumerate(loop):
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        
        optimizer.zero_grad()
        x_reconstructed, mu, sigma = model(x)
        
        reconstruction_loss = recon_loss_fn(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2) + 1e-8) - mu.pow(2) - sigma.pow(2))
        
        total_loss = reconstruction_loss + beta * kl_divergence
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        loop.set_postfix(loss=total_loss.item())
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    model.eval()
    with torch.no_grad():
        sample_z = torch.randn(8, Z_DIM).to(DEVICE)
        model.original_size = (8, 3, 64, 64)
        samples = model.decode(sample_z)
        
        rand_idx = np.random.randint(0, len(dataset))
        real_img, _ = dataset[rand_idx]
        real_img = real_img.unsqueeze(0).to(DEVICE)
        recon_img, _, _ = model(real_img.view(1, -1))
        recon_img = recon_img.view(1, 3, 64, 64)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        real_display = real_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        recon_display = recon_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        axs[0].imshow(np.clip(real_display, 0, 1))
        axs[0].set_title("Original")
        axs[0].axis("off")
        
        axs[1].imshow(np.clip(recon_display, 0, 1))
        axs[1].set_title("Reconstruction")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"reconstruction_epoch_{epoch+1}.png")
        plt.close()

torch.save(model.state_dict(), "improved_vae_model.pt")

def inference(model, target_size=(64, 64)):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, Z_DIM).to(DEVICE)
        model.original_size = (1, 3, target_size[0], target_size[1])
        generated_image = model.decode(z)
        
        generated_image = generated_image.view(1, 3, target_size[0], target_size[1])
        generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        generated_image = np.clip(generated_image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(generated_image)

image = inference(model)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.title("Generated Image")
plt.show()
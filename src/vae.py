import torch
import torch.nn as nn

# from encoder import VAE_Encoder
# from decoder import VAE_Decoder



# class VAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = VAE_Encoder()
#         self.decoder = VAE_Decoder()

#     def forward(self, x):
#         batch_size, _, height, width = x.shape

#         noise = torch.randn(batch_size, 4, height // 8, width // 8).to(x.device)

#         latents = self.encoder(x, noise)

#         reconstructed = self.decoder(latents)

#         return reconstructed, latents, noise
        
    
#     def loss_function(self, recon_x, x, latents, noise):
#         # Reconstruction loss
#         recon_loss = nn.functional.mse_loss(recon_x, x)

#         # KL divergence loss
#         kl_loss = -0.5 * torch.sum(1 + noise - noise.pow(2) - noise.exp())

#         return recon_loss + 0.0001 * kl_loss, recon_loss, kl_loss



class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_2_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_2_sigma = nn.Linear(hidden_dim, z_dim)
        
        #decoder
        self.z_2_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_2_img = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        
        # Store dimensions for potential inference
        self.input_dim = input_dim
        self.z_dim = z_dim
        
    def encode(self, x):
        # If x is not flattened, flatten it
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
            
        # q_phi(z given x)
        h = self.img_2_hidden(x)
        h = self.relu(h)
        
        mu, sigma = self.hidden_2_mu(h), self.hidden_2_sigma(h)
        
        return mu, sigma

    def decode(self, z):
        # p_theta(x given z)
        h = self.z_2_hidden(z)
        h = self.relu(h)
        h = self.hidden_2_img(h)
        h = torch.sigmoid(h) # for binary cross entropy loss
        
        return h
        
    def forward(self, x):
        # Handle either flattened or unflattened input
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(original_shape[0], -1)
            
        mu, sigma = self.encode(x)
        
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        
        return x_reconstructed, mu, sigma
    
    # Method for generating new images during inference
    def sample(self, num_samples=1, device="cpu"):
        z = torch.randn(num_samples, self.z_dim).to(device)
        samples = self.decode(z)
        return samples
    


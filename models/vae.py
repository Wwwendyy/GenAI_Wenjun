import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),               # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),              # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),             # 8 -> 4
            nn.ReLU(),
        )

        self.feature_dim = 256 * 4 * 4

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),      # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = mu + sigma * eps
        sigma = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), 256, 4, 4)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss_function(x_recon, x, mu, logvar, beta=1.0, recon_loss_type="bce"):
    batch_size = x.size(0)

    if recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / batch_size
    elif recon_loss_type == "mse":
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
    else:
        raise ValueError("recon_loss_type must be 'bce' or 'mse'")

    # KL divergence: D_KL(q(z|x) || p(z))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
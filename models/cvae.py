import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, num_classes=102, image_channels=3, latent_dim=128, image_size=64, label_emb_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb_dim = label_emb_dim
        self.image_size = image_size

        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)

        # Encoder input: image + label map
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.feature_dim = 256 * 4 * 4

        self.fc_mu = nn.Linear(self.feature_dim + label_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim + label_emb_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim + label_emb_dim, self.feature_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def make_label_map(self, y, image_size):
        """
        把 label 变成 (B,1,H,W) 的条件图
        """
        label_map = y.float().view(-1, 1, 1, 1)
        label_map = label_map / (self.num_classes - 1)
        label_map = label_map.expand(-1, 1, image_size, image_size)
        return label_map

    def encode(self, x, y):
        label_map = self.make_label_map(y, x.size(2)).to(x.device)
        x_cond = torch.cat([x, label_map], dim=1)

        h = self.encoder(x_cond)
        h = h.view(x.size(0), -1)

        y_emb = self.label_embedding(y)
        h = torch.cat([h, y_emb], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_embedding(y)
        z_cond = torch.cat([z, y_emb], dim=1)

        h = self.fc_decode(z_cond)
        h = h.view(z.size(0), 256, 4, 4)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar


def cvae_loss_function(x_recon, x, mu, logvar, beta=1.0, recon_loss_type="bce"):
    batch_size = x.size(0)

    if recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / batch_size
    elif recon_loss_type == "mse":
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
    else:
        raise ValueError("recon_loss_type must be 'bce' or 'mse'")

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
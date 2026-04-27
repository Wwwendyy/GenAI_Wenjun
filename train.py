import os
import argparse
import torch
import torch.optim as optim

from utils import get_dataloaders, save_generated_images
from models.vae import VAE, vae_loss_function
from models.cvae import CVAE, cvae_loss_function


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = get_dataloaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes)
    print("Class to idx:", train_dataset.class_to_idx)

    if args.conditional:
        model = CVAE(
            num_classes=num_classes,
            image_channels=3,
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            label_emb_dim=args.label_emb_dim
        ).to(device)
    else:
        model = VAE(
            image_channels=3,
            latent_dim=args.latent_dim,
            image_size=args.image_size
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss_epoch = 0.0
        total_recon_epoch = 0.0
        total_kl_epoch = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.conditional:
                recon, mu, logvar = model(images, labels)
                loss, recon_loss, kl_loss = cvae_loss_function(
                    recon, images, mu, logvar,
                    beta=args.beta,
                    recon_loss_type=args.recon_loss_type
                )
            else:
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon, images, mu, logvar,
                    beta=args.beta,
                    recon_loss_type=args.recon_loss_type
                )

            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_recon_epoch += recon_loss.item()
            total_kl_epoch += kl_loss.item()

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Recon: {recon_loss.item():.4f} "
                    f"KL: {kl_loss.item():.4f}"
                )

        avg_loss = total_loss_epoch / len(train_loader)
        avg_recon = total_recon_epoch / len(train_loader)
        avg_kl = total_kl_epoch / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Avg Loss:  {avg_loss:.4f}")
        print(f"Avg Recon: {avg_recon:.4f}")
        print(f"Avg KL:    {avg_kl:.4f}\n")

        # 每个 epoch 保存一些随机生成图
        model.eval()
        with torch.no_grad():
            z = torch.randn(16, args.latent_dim).to(device)

            if args.conditional:
                sample_labels = torch.arange(16).to(device) % num_classes
                samples = model.decode(z, sample_labels)
            else:
                samples = model.decode(z)

            save_generated_images(
                samples,
                os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.png"),
                nrow=4
            )

        torch.save(
            model.state_dict(),
            os.path.join(args.ckpt_dir, f"model_epoch_{epoch+1:03d}.pth")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/flowers")
    parser.add_argument("--output_dir", type=str, default="./outputs/samples")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs/checkpoints")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--label_emb_dim", type=int, default=32)
    parser.add_argument("--recon_loss_type", type=str, default="mse", choices=["bce", "mse"])

    args = parser.parse_args()
    train(args)
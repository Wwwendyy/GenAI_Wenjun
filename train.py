# train.py
import os
import csv
import argparse
import torch
import torch.optim as optim

from utils import get_dataloaders, save_generated_images
from models.vae import VAE, vae_loss_function
from models.cvae import CVAE, cvae_loss_function


def compute_beta(args, epoch):
    if not args.kl_annealing:
        return args.beta

    warmup = max(1, args.kl_warmup_epochs)
    beta = args.beta * min(1.0, (epoch + 1) / warmup)
    return beta


def evaluate(model, valid_loader, device, args, current_beta):
    model.eval()

    valid_loss_epoch = 0.0
    valid_recon_epoch = 0.0
    valid_kl_epoch = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            if args.conditional:
                recon, mu, logvar = model(images, labels)
                loss, recon_loss, kl_loss = cvae_loss_function(
                    recon,
                    images,
                    mu,
                    logvar,
                    beta=current_beta,
                    recon_loss_type=args.recon_loss_type
                )
            else:
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon,
                    images,
                    mu,
                    logvar,
                    beta=current_beta,
                    recon_loss_type=args.recon_loss_type
                )

            valid_loss_epoch += loss.item()
            valid_recon_epoch += recon_loss.item()
            valid_kl_epoch += kl_loss.item()

    avg_valid_loss = valid_loss_epoch / len(valid_loader)
    avg_valid_recon = valid_recon_epoch / len(valid_loader)
    avg_valid_kl = valid_kl_epoch / len(valid_loader)

    return avg_valid_loss, avg_valid_recon, avg_valid_kl


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create separated experiment folders
    exp_dir = os.path.join(args.base_output_dir, args.exp_name)
    sample_dir = os.path.join(exp_dir, "samples")
    recon_dir = os.path.join(exp_dir, "reconstructions")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "training_log.csv")

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = get_dataloaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    num_classes = len(train_dataset.classes)

    print("Experiment:", args.exp_name)
    print("Experiment dir:", exp_dir)
    print("Data root:", args.data_dir)
    print("Train size:", len(train_dataset))
    print("Valid size:", len(valid_dataset))

    if test_dataset is not None:
        print("Test size:", len(test_dataset))
    else:
        print("Test size: No test folder found")

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

    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "beta",
            "train_loss",
            "train_recon",
            "train_kl",
            "valid_loss",
            "valid_recon",
            "valid_kl"
        ])

    for epoch in range(args.epochs):
        model.train()

        current_beta = compute_beta(args, epoch)

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
                    recon,
                    images,
                    mu,
                    logvar,
                    beta=current_beta,
                    recon_loss_type=args.recon_loss_type
                )
            else:
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon,
                    images,
                    mu,
                    logvar,
                    beta=current_beta,
                    recon_loss_type=args.recon_loss_type
                )

            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_recon_epoch += recon_loss.item()
            total_kl_epoch += kl_loss.item()

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Beta: {current_beta:.4f} "
                    f"Loss: {loss.item():.4f} "
                    f"Recon: {recon_loss.item():.4f} "
                    f"KL: {kl_loss.item():.4f}"
                )

        avg_loss = total_loss_epoch / len(train_loader)
        avg_recon = total_recon_epoch / len(train_loader)
        avg_kl = total_kl_epoch / len(train_loader)

        avg_valid_loss, avg_valid_recon, avg_valid_kl = evaluate(
            model,
            valid_loader,
            device,
            args,
            current_beta
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Beta:        {current_beta:.4f}")
        print(f"Train Loss:  {avg_loss:.4f}")
        print(f"Train Recon: {avg_recon:.4f}")
        print(f"Train KL:    {avg_kl:.4f}")
        print(f"Valid Loss:  {avg_valid_loss:.4f}")
        print(f"Valid Recon: {avg_valid_recon:.4f}")
        print(f"Valid KL:    {avg_valid_kl:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_beta,
                avg_loss,
                avg_recon,
                avg_kl,
                avg_valid_loss,
                avg_valid_recon,
                avg_valid_kl
            ])

        # Save random generated samples
        model.eval()
        with torch.no_grad():
            z = torch.randn(16, args.latent_dim).to(device)

            if args.conditional:
                sample_labels = torch.arange(16, device=device) % num_classes
                samples = model.decode(z, sample_labels)
            else:
                samples = model.decode(z)

            save_generated_images(
                samples,
                os.path.join(sample_dir, f"samples_epoch_{epoch + 1:03d}.png"),
                nrow=4
            )

            # Save reconstruction comparison
            real_images, real_labels = next(iter(valid_loader))
            real_images = real_images[:8].to(device)
            real_labels = real_labels[:8].to(device)

            if args.conditional:
                recon_images, _, _ = model(real_images, real_labels)
            else:
                recon_images, _, _ = model(real_images)

            comparison = torch.cat([real_images, recon_images], dim=0)

            save_generated_images(
                comparison,
                os.path.join(recon_dir, f"recon_epoch_{epoch + 1:03d}.png"),
                nrow=8
            )

        # Save latest checkpoint
        latest_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Early stopping + best checkpoint
        if avg_valid_loss < best_valid_loss - args.min_delta:
            best_valid_loss = avg_valid_loss
            epochs_without_improvement = 0

            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{args.patience} epochs"
            )

        print(f"Saved outputs under: {exp_dir}\n")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            print(f"Best validation loss: {best_valid_loss:.4f}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./flower_dataset/dataset")

    # New output organization
    parser.add_argument("--base_output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--label_emb_dim", type=int, default=32)

    parser.add_argument(
        "--recon_loss_type",
        type=str,
        default="mse",
        choices=["bce", "mse"]
    )

    # Early stopping
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    # Checkpoint saving
    parser.add_argument("--save_every", type=int, default=10)

    # KL annealing
    parser.add_argument("--kl_annealing", action="store_true")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)

    args = parser.parse_args()
    train(args)
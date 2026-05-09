# sample.py
import os
import argparse
import torch

from utils import save_generated_images, interpolate_latent
from models.vae import VAE
from models.cvae import CVAE


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # New separated output folder:
    # outputs/<exp_name>/generated/
    output_dir = os.path.join(args.base_output_dir, args.exp_name, "generated")
    os.makedirs(output_dir, exist_ok=True)

    if args.conditional:
        model = CVAE(
            num_classes=args.num_classes,
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

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # Random samples
        z = torch.randn(args.num_samples, args.latent_dim).to(device)

        if args.conditional:
            labels = torch.randint(
                low=0,
                high=args.num_classes,
                size=(args.num_samples,),
                device=device
            )
            samples = model.decode(z, labels)
        else:
            samples = model.decode(z)

        random_path = os.path.join(output_dir, "random_samples.png")
        save_generated_images(samples, random_path, nrow=4)
        print(f"Saved random samples to: {random_path}")

        # Class-fixed samples for CVAE
        if args.conditional:
            fixed_z = torch.randn(args.num_samples, args.latent_dim).to(device)
            fixed_labels = torch.tensor(
                [args.class_id] * args.num_samples,
                device=device
            )
            class_samples = model.decode(fixed_z, fixed_labels)

            class_path = os.path.join(output_dir, f"class_{args.class_id}_samples.png")
            save_generated_images(class_samples, class_path, nrow=4)
            print(f"Saved class-conditioned samples to: {class_path}")

        # Latent interpolation
        z1 = torch.randn(args.latent_dim).to(device)
        z2 = torch.randn(args.latent_dim).to(device)

        if args.conditional:
            interp_labels = torch.tensor(
                [args.class_id] * args.interp_steps,
                device=device
            )
            interp_imgs = interpolate_latent(
                model,
                z1,
                z2,
                steps=args.interp_steps,
                device=device,
                labels=interp_labels
            )
        else:
            interp_imgs = interpolate_latent(
                model,
                z1,
                z2,
                steps=args.interp_steps,
                device=device
            )

        interp_path = os.path.join(output_dir, "interpolation.png")
        save_generated_images(interp_imgs, interp_path, nrow=args.interp_steps)
        print(f"Saved interpolation to: {interp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)

    # Instead of directly writing to outputs/test_samples,
    # this creates outputs/<exp_name>/generated/
    parser.add_argument("--base_output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=16)

    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--num_classes", type=int, default=102)
    parser.add_argument("--label_emb_dim", type=int, default=32)
    parser.add_argument("--class_id", type=int, default=0)

    parser.add_argument("--interp_steps", type=int, default=10)

    args = parser.parse_args()
    sample(args)
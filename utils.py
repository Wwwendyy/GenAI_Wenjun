import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


def get_transforms(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),   # [0,1]
    ])


def get_dataloaders(data_root, batch_size=64, image_size=64, num_workers=2):
    """
    data_root:
        ./data/dataset

    expects:
        data_root/train/class_x/xxx.jpg
        data_root/valid/class_x/xxx.jpg
        data_root/test/class_x/xxx.jpg
    """
    transform = get_transforms(image_size)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    test_dir  = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def save_generated_images(images, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(images, path, nrow=nrow)


def show_images(images, nrow=8, title=None):
    grid = make_grid(images, nrow=nrow)
    np_img = grid.detach().cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def interpolate_latent(model, z1, z2, steps=10, device="cpu", labels=None):
    z_list = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        z_list.append(z)

    z_batch = torch.stack(z_list, dim=0).to(device)

    with torch.no_grad():
        if labels is not None:
            recon = model.decode(z_batch, labels)
        else:
            recon = model.decode(z_batch)

    return recon
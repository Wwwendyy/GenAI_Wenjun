# utils.py
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FlatImageDataset(Dataset):
    """
    Dataset for folders that directly contain images, without class subfolders.

    Example:
        test/
            image_00001.jpg
            image_00002.jpg
            ...
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.image_paths = [
            os.path.join(root, fname)
            for fname in sorted(os.listdir(root))
            if fname.lower().endswith(IMG_EXTENSIONS)
        ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in flat folder: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # test set has no label, so return dummy label 0
        return image, 0


def get_transforms(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def has_class_subfolders(folder):
    """
    Check whether a folder contains class subfolders.
    ImageFolder needs at least one subfolder.
    """
    if not os.path.isdir(folder):
        return False

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            return True

    return False


def get_dataloaders(data_root, batch_size=64, image_size=64, num_workers=2):
    """
    Expected structure for train/valid:

    data_root/
        train/
            class_1/
            class_2/
            ...
        valid/
            class_1/
            class_2/
            ...

    test can be either:

    data_root/
        test/
            class_1/
            class_2/
            ...

    or:

    data_root/
        test/
            image_00001.jpg
            image_00002.jpg
            ...

    For your current project:
        data_root = ./flower_dataset/dataset
    """
    transform = get_transforms(image_size)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    if not os.path.isdir(valid_dir):
        raise FileNotFoundError(f"Valid folder not found: {valid_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

    # Your Kaggle test folder is flat, so ImageFolder cannot read it.
    if os.path.isdir(test_dir):
        if has_class_subfolders(test_dir):
            test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        else:
            test_dataset = FlatImageDataset(test_dir, transform=transform)
    else:
        test_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        test_loader = None

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def save_generated_images(images, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = torch.clamp(images, 0, 1)
    save_image(images, path, nrow=nrow)


def show_images(images, nrow=8, title=None):
    images = torch.clamp(images, 0, 1)
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

    for alpha in torch.linspace(0, 1, steps, device=device):
        z = (1 - alpha) * z1 + alpha * z2
        z_list.append(z)

    z_batch = torch.stack(z_list, dim=0).to(device)

    with torch.no_grad():
        if labels is not None:
            labels = labels.to(device)
            recon = model.decode(z_batch, labels)
        else:
            recon = model.decode(z_batch)

    return recon
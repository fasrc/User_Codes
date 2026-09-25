#!/usr/bin/env python3

import argparse
import gzip
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Pure-PyTorch MNIST loader — no torchvision dependency
# ---------------------------------------------------------------------------

MNIST_URLS = {
    "train_images":
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
        "train-images-idx3-ubyte.gz",

    "train_labels":
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
        "train-labels-idx1-ubyte.gz",

    "test_images":
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
        "t10k-images-idx3-ubyte.gz",

    "test_labels":
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
        "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(root="./data"):
    os.makedirs(root, exist_ok=True)

    for name, url in MNIST_URLS.items():
        path = os.path.join(root, name + ".gz")

        if not os.path.exists(path):
            print(f"Downloading {name}...", flush=True)
            urllib.request.urlretrieve(url, path)


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(
                f"Invalid image file magic number in {path}: {magic}"
            )

        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data.reshape(n, h, w)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(
                f"Invalid label file magic number in {path}: {magic}"
            )

        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data


class MNISTDataset(Dataset):
    """
    Minimal MNIST dataset without torchvision.
    """

    MEAN = 0.1307
    STD = 0.3081

    def __init__(self, root="./data", train=True):

        if train:
            images = read_images(
                os.path.join(root, "train_images.gz")
            )

            labels = read_labels(
                os.path.join(root, "train_labels.gz")
            )

        else:
            images = read_images(
                os.path.join(root, "test_images.gz")
            )

            labels = read_labels(
                os.path.join(root, "test_labels.gz")
            )

        x = images.astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD

        self.images = torch.from_numpy(x).unsqueeze(1)

        self.labels = torch.from_numpy(
            labels.astype(np.int64)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)

        return self.fc(x)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested_device):

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() "
                "is False."
            )

        return torch.device("cuda:0")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
):
    """
    Save enough state to resume basic training.

    This intentionally includes only:
      - completed epoch
      - model parameters
      - optimizer state

    More complete training state will be added in Part C.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Write to a temporary file first, then atomically replace
    # the previous checkpoint.
    tmp_path = checkpoint_path + ".tmp"

    torch.save(checkpoint, tmp_path)

    os.replace(tmp_path, checkpoint_path)

    print(
        f"Checkpoint saved: {checkpoint_path} "
        f"(after epoch {epoch})",
        flush=True,
    )


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
):
    """
    Restore the model and optimizer and return the next epoch.
    """

    print(
        f"Loading checkpoint: {checkpoint_path}",
        flush=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = checkpoint["epoch"]

    start_epoch = completed_epoch + 1

    print(
        f"Checkpoint restored from epoch {completed_epoch}.",
        flush=True,
    )

    print(
        f"Training will resume at epoch {start_epoch}.",
        flush=True,
    )

    return start_epoch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
):

    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        batch_size = data.size(0)

        loss_sum += loss.item() * batch_size

        correct += (
            output.argmax(dim=1)
            .eq(target)
            .sum()
            .item()
        )

        total += batch_size

        if batch_idx % 100 == 0:

            print(
                f"  Epoch {epoch} "
                f"| batch {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f}",
                flush=True,
            )

    train_loss = loss_sum / total
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for data, target in loader:

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = criterion(output, target)

            batch_size = data.size(0)

            loss_sum += loss.item() * batch_size

            correct += (
                output.argmax(dim=1)
                .eq(target)
                .sum()
                .item()
            )

            total += batch_size

    val_loss = loss_sum / total
    val_accuracy = 100.0 * correct / total

    return val_loss, val_accuracy


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="MNIST training with basic checkpoint/restart"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mnist_checkpoint.pt",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each epoch",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Training device. "
            "'auto' uses CUDA when available, otherwise CPU."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    device = select_device(args.device)

    print("============================================================")
    print("PyTorch MNIST — basic checkpoint/restart")
    print("============================================================")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")

    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(device)}")

    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Resume          : {args.resume}")
    print("============================================================")
    print(flush=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    download_mnist(args.data_root)

    train_dataset = MNISTDataset(
        root=args.data_root,
        train=True,
    )

    val_dataset = MNISTDataset(
        root=args.data_root,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # -----------------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------------

    model = MNISTNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------------------------------------------------------
    # Restore checkpoint if requested
    # -----------------------------------------------------------------------

    start_epoch = 1

    if args.resume:

        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}"
            )

        start_epoch = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device,
        )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    start_time = time.time()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f" | train loss {train_loss:.4f}"
            f" | train acc {train_accuracy:.2f}%"
            f" | val loss {val_loss:.4f}"
            f" | val acc {val_accuracy:.2f}%"
            f" | time {epoch_time:.2f} s",
            flush=True,
        )

        # Save after every completed epoch.
        save_checkpoint(
            args.checkpoint,
            epoch,
            model,
            optimizer,
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    total_time = time.time() - start_time

    print()
    print("============================================================")
    print("Training complete")
    print("============================================================")
    print(f"Total time : {total_time:.2f} s")
    print(f"Checkpoint : {args.checkpoint}")
    print("============================================================")


if __name__ == "__main__":
    main()

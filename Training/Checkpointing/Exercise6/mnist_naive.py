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

    Images are normalized using the standard MNIST mean and standard
    deviation.
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
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
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
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="CPU MNIST training without checkpointing"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size",
    )

    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=512,
        help="Validation batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="MNIST data directory",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader workers",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help=(
            "Seconds to sleep after each epoch. "
            "Useful for checkpoint/restart demonstrations."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    # This exercise intentionally uses CPU training.
    device = torch.device("cpu")

    print("============================================================")
    print("PyTorch MNIST — baseline CPU training")
    print("============================================================")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Workers         : {args.workers}")
    print("Checkpointing   : disabled")
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
    # Model
    # -----------------------------------------------------------------------

    model = MNISTNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

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

        if args.sleep > 0:
            time.sleep(args.sleep)

    total_time = time.time() - start_time

    print()
    print("============================================================")
    print("Training complete")
    print("============================================================")
    print(f"Total time : {total_time:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    main()

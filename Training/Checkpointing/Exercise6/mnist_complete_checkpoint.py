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
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
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
# Optimizer helper
# ---------------------------------------------------------------------------

def move_optimizer_state_to_device(optimizer, device):
    """
    Move tensor-valued optimizer state to the selected device.

    The checkpoint is always deserialized onto CPU first for portability.
    After optimizer.load_state_dict(), this function ensures that Adam's
    internal tensor state is placed on the same device as the model.
    """

    for state in optimizer.state.values():

        for key, value in state.items():

            if torch.is_tensor(value):
                state[key] = value.to(device)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    early_stopping_counter,
    train_generator,
    device,
):

    checkpoint = {
        "version": 1,

        # Last fully completed epoch.
        "epoch": epoch,

        # Model state.
        "model_state_dict": model.state_dict(),

        # Optimizer state.
        "optimizer_state_dict": optimizer.state_dict(),

        # Learning-rate scheduler state.
        "scheduler_state_dict": scheduler.state_dict(),

        # Early-stopping state.
        "best_val_loss": best_val_loss,
        "early_stopping_counter": early_stopping_counter,

        # CPU RNG state.
        "torch_rng_state": torch.get_rng_state(),

        # RNG used by DataLoader shuffling.
        "train_generator_state": train_generator.get_state(),

        # Record where the training process was running.
        "saved_device_type": device.type,
    }

    # If training is running on CUDA, preserve the RNG state for the
    # particular CUDA device being used.
    if device.type == "cuda":

        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state(device).cpu()
        )

    # -----------------------------------------------------------------------
    # Atomic checkpoint replacement
    # -----------------------------------------------------------------------

    tmp_path = checkpoint_path + ".tmp"

    with open(tmp_path, "wb") as f:

        torch.save(checkpoint, f)

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp_path,
        checkpoint_path,
    )

    print(
        f"Checkpoint saved: {checkpoint_path} "
        f"(after epoch {epoch})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    train_generator,
    device,
):

    print(
        f"Loading checkpoint: {checkpoint_path}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # Always deserialize onto CPU first.
    #
    # This keeps CPU RNG state as a CPU ByteTensor and also makes the
    # checkpoint portable between CPU and GPU runs.
    # -----------------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # model already lives on the requested device, and load_state_dict()
    # copies the saved parameter values into the model parameters.

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Because the checkpoint was loaded onto CPU, explicitly move any
    # optimizer tensors (e.g. Adam running moments) to the model device.
    move_optimizer_state_to_device(
        optimizer,
        device,
    )

    # -----------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # -----------------------------------------------------------------------
    # Early-stopping state
    # -----------------------------------------------------------------------

    best_val_loss = checkpoint["best_val_loss"]

    early_stopping_counter = (
        checkpoint["early_stopping_counter"]
    )

    # -----------------------------------------------------------------------
    # CPU RNG state
    # -----------------------------------------------------------------------

    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    # -----------------------------------------------------------------------
    # DataLoader shuffle RNG
    # -----------------------------------------------------------------------

    train_generator.set_state(
        checkpoint["train_generator_state"].cpu()
    )

    # -----------------------------------------------------------------------
    # CUDA RNG state
    # -----------------------------------------------------------------------

    saved_device_type = checkpoint.get(
        "saved_device_type",
        "unknown",
    )

    if device.type == "cuda":

        if "cuda_rng_state" in checkpoint:

            torch.cuda.set_rng_state(
                checkpoint["cuda_rng_state"].cpu(),
                device=device,
            )

            print(
                "CUDA RNG state restored.",
                flush=True,
            )

        else:

            print(
                "Note: checkpoint has no CUDA RNG state; "
                "CUDA training can continue, but the random "
                "sequence cannot exactly continue from the "
                "previous run.",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Restart location
    # -----------------------------------------------------------------------

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

    print(
        f"Checkpoint was saved from device type: "
        f"{saved_device_type}",
        flush=True,
    )

    print(
        f"Current device: {device}",
        flush=True,
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}",
        flush=True,
    )

    print(
        f"Early-stopping counter: "
        f"{early_stopping_counter}",
        flush=True,
    )

    print(
        f"Current learning rate: "
        f"{optimizer.param_groups[0]['lr']:.6g}",
        flush=True,
    )

    return (
        start_epoch,
        best_val_loss,
        early_stopping_counter,
    )


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

    use_cuda = device.type == "cuda"

    for batch_idx, (data, target) in enumerate(loader):

        data = data.to(
            device,
            non_blocking=use_cuda,
        )

        target = target.to(
            device,
            non_blocking=use_cuda,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(data)

        loss = criterion(
            output,
            target,
        )

        loss.backward()

        optimizer.step()

        batch_size = data.size(0)

        loss_sum += (
            loss.detach().item()
            * batch_size
        )

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

    train_loss = (
        loss_sum / total
    )

    train_accuracy = (
        100.0 * correct / total
    )

    return (
        train_loss,
        train_accuracy,
    )


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

    use_cuda = device.type == "cuda"

    with torch.no_grad():

        for data, target in loader:

            data = data.to(
                device,
                non_blocking=use_cuda,
            )

            target = target.to(
                device,
                non_blocking=use_cuda,
            )

            output = model(data)

            loss = criterion(
                output,
                target,
            )

            batch_size = data.size(0)

            loss_sum += (
                loss.detach().item()
                * batch_size
            )

            correct += (
                output.argmax(dim=1)
                .eq(target)
                .sum()
                .item()
            )

            total += batch_size

    val_loss = (
        loss_sum / total
    )

    val_accuracy = (
        100.0 * correct / total
    )

    return (
        val_loss,
        val_accuracy,
    )


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "MNIST training with complete "
            "CPU/GPU checkpoint and restart"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of epochs",
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
        help="Initial learning rate",
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
        "--checkpoint",
        type=str,
        default="mnist_complete.pt",
        help="Checkpoint file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Training device. "
            "'auto' uses CUDA when available, "
            "otherwise CPU."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help=(
            "Epochs without validation-loss improvement "
            "before reducing the learning rate"
        ),
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Epochs without validation-loss improvement "
            "before stopping training"
        ),
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

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    device = select_device(
        args.device
    )

    use_cuda = (
        device.type == "cuda"
    )

    if use_cuda:
        torch.cuda.set_device(device)

    # -----------------------------------------------------------------------
    # Initial RNG seeds
    #
    # These establish deterministic starting states for a fresh run.
    # When --resume is used, the saved states will later overwrite them.
    # -----------------------------------------------------------------------

    torch.manual_seed(
        args.seed
    )

    if use_cuda:
        torch.cuda.manual_seed_all(
            args.seed
        )

    # A separate CPU generator controls training-data shuffling.
    train_generator = torch.Generator(
        device="cpu"
    )

    train_generator.manual_seed(
        args.seed
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    print(
        "============================================================"
    )
    print(
        "PyTorch MNIST — complete training checkpoint"
    )
    print(
        "============================================================"
    )

    print(
        f"PyTorch version       : "
        f"{torch.__version__}"
    )

    print(
        f"Requested device      : "
        f"{args.device}"
    )

    print(
        f"Selected device       : "
        f"{device}"
    )

    if use_cuda:

        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Epochs                : "
        f"{args.epochs}"
    )

    print(
        f"Batch size            : "
        f"{args.batch_size}"
    )

    print(
        f"Initial learning rate : "
        f"{args.lr}"
    )

    print(
        f"LR patience           : "
        f"{args.lr_patience}"
    )

    print(
        f"Early-stop patience   : "
        f"{args.early_stopping_patience}"
    )

    print(
        f"Checkpoint            : "
        f"{args.checkpoint}"
    )

    print(
        f"Resume                : "
        f"{args.resume}"
    )

    print(
        f"Seed                  : "
        f"{args.seed}"
    )

    print(
        "============================================================"
    )

    print(flush=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    download_mnist(
        args.data_root
    )

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
        generator=train_generator,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = MNISTNet().to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # Scheduler must already exist before optimizer checkpoint state
    # is restored.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
    )

    # -----------------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------------

    start_epoch = 1

    best_val_loss = float(
        "inf"
    )

    early_stopping_counter = 0

    # -----------------------------------------------------------------------
    # Restore checkpoint
    # -----------------------------------------------------------------------

    if args.resume:

        if not os.path.exists(
            args.checkpoint
        ):
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{args.checkpoint}"
            )

        (
            start_epoch,
            best_val_loss,
            early_stopping_counter,
        ) = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            scheduler,
            train_generator,
            device,
        )

        if start_epoch > args.epochs:

            print(
                f"Checkpoint already contains epoch "
                f"{start_epoch - 1}, which is at or beyond "
                f"the requested --epochs {args.epochs}.",
                flush=True,
            )

            return

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    start_time = time.time()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
        )

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------

        (
            val_loss,
            val_accuracy,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # ---------------------------------------------------------------
        # Learning-rate scheduler
        # ---------------------------------------------------------------

        old_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        scheduler.step(
            val_loss
        )

        new_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        if new_lr != old_lr:

            print(
                f"Learning rate reduced: "
                f"{old_lr:.6g} -> "
                f"{new_lr:.6g}",
                flush=True,
            )

        # ---------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            early_stopping_counter = 0

            improvement = "yes"

        else:

            early_stopping_counter += 1

            improvement = "no"

        # ---------------------------------------------------------------
        # Epoch summary
        # ---------------------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f" | train loss {train_loss:.4f}"
            f" | train acc {train_accuracy:.2f}%"
            f" | val loss {val_loss:.4f}"
            f" | val acc {val_accuracy:.2f}%"
            f" | lr {new_lr:.6g}"
            f" | improved {improvement}"
            f" | patience "
            f"{early_stopping_counter}/"
            f"{args.early_stopping_patience}"
            f" | time {epoch_time:.2f} s",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Checkpoint
        #
        # Save only after all state associated with this epoch has been
        # updated. The checkpoint therefore represents a fully completed
        # epoch.
        # ---------------------------------------------------------------

        save_checkpoint(
            args.checkpoint,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            early_stopping_counter,
            train_generator,
            device,
        )

        # ---------------------------------------------------------------
        # Early-stopping decision
        # ---------------------------------------------------------------

        if (
            early_stopping_counter
            >= args.early_stopping_patience
        ):

            print()

            print(
                "Early stopping triggered.",
                flush=True,
            )

            print(
                f"No validation-loss improvement "
                f"for {early_stopping_counter} "
                f"consecutive epochs.",
                flush=True,
            )

            break

        # Artificial pause for workshop demonstrations.
        if args.sleep > 0:

            time.sleep(
                args.sleep
            )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print()

    print(
        "============================================================"
    )

    print(
        "Training finished"
    )

    print(
        "============================================================"
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Final learning rate  : "
        f"{optimizer.param_groups[0]['lr']:.6g}"
    )

    print(
        f"Total time           : "
        f"{total_time:.2f} s"
    )

    print(
        f"Checkpoint           : "
        f"{args.checkpoint}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()

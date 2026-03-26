import datetime
import os
import gzip
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ---------------------------------------------------------------------------
# Pure-PyTorch MNIST loader — no torchvision dependency
# ---------------------------------------------------------------------------
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(root="./data"):
    os.makedirs(root, exist_ok=True)
    for name, url in MNIST_URLS.items():
        fpath = os.path.join(root, name + ".gz")
        if not os.path.exists(fpath):
            print(f"Downloading {name}...", flush=True)
            urllib.request.urlretrieve(url, fpath)


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h, w)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


class MNISTDataset(Dataset):
    """
    Minimal MNIST dataset — no torchvision required.
    Normalizes to zero mean / unit std using standard MNIST statistics.
    """
    MEAN = 0.1307
    STD = 0.3081

    def __init__(self, root="./data", train=True):
        if train:
            images = read_images(os.path.join(root, "train_images.gz"))
            labels = read_labels(os.path.join(root, "train_labels.gz"))
        else:
            images = read_images(os.path.join(root, "test_images.gz"))
            labels = read_labels(os.path.join(root, "test_labels.gz"))

        x = images.astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD

        self.images = torch.from_numpy(x).unsqueeze(1)   # (N, 1, 28, 28)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model: simple CNN for MNIST
# ---------------------------------------------------------------------------
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
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
# Setup and teardown
# ---------------------------------------------------------------------------
def setup():
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Build the store explicitly with the IPv4 address — bypasses getaddrinfo IPv6
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]   # already an IPv4 from the batch script
    master_port = int(os.environ["MASTER_PORT"])

    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=120),
    )
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distributed metric helpers
# ---------------------------------------------------------------------------
def reduce_sum(value, device):
    """
    All-reduce a scalar value across all ranks and return the summed result.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
    else:
        tensor = torch.tensor(value, dtype=torch.float64, device=device)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, rank):
    model.train()
    loader.sampler.set_epoch(epoch)

    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        local_loss_sum += loss.item() * batch_size
        local_correct += output.argmax(1).eq(target).sum().item()
        local_total += batch_size

        if rank == 0 and batch_idx % 100 == 0:
            print(
                f"  Epoch {epoch} | batch {batch_idx}/{len(loader)} "
                f"| local loss {loss.item():.4f}",
                flush=True,
            )

    global_loss_sum = reduce_sum(local_loss_sum, device).item()
    global_correct = reduce_sum(local_correct, device).item()
    global_total = reduce_sum(local_total, device).item()

    train_loss = global_loss_sum / global_total
    train_acc = 100.0 * global_correct / global_total

    return train_loss, train_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            batch_size = data.size(0)
            local_loss_sum += loss.item() * batch_size
            local_correct += output.argmax(1).eq(target).sum().item()
            local_total += batch_size

    global_loss_sum = reduce_sum(local_loss_sum, device).item()
    global_correct = reduce_sum(local_correct, device).item()
    global_total = reduce_sum(local_total, device).item()

    val_loss = global_loss_sum / global_total
    val_acc = 100.0 * global_correct / global_total

    return val_loss, val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    if rank == 0:
        print(f"Training with {world_size} GPUs", flush=True)
        print(f"Using {num_workers} DataLoader worker(s) per rank", flush=True)

    # Download only on rank 0, then wait for all ranks.
    if rank == 0:
        download_mnist("./data")
    dist.barrier()

    train_dataset = MNISTDataset(root="./data", train=True)
    val_dataset = MNISTDataset(root="./data", train=False)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = MNISTNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if rank == 0:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f}  acc {val_acc:.2f}%",
                flush=True,
            )

    dist.barrier()

    if rank == 0:
        torch.save(model.module.state_dict(), "mnist_ddp.pt")
        print("Checkpoint saved to mnist_ddp.pt", flush=True)

    cleanup()


if __name__ == "__main__":
    main()

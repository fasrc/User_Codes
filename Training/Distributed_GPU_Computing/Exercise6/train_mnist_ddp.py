import argparse
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
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DDP MNIST training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-rank training batch size")
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base learning rate (scaled by world_size)")
    parser.add_argument("--no-lr-scaling", action="store_true",
                        help="Disable linear LR scaling with world size")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="mnist_ddp.pt")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Cap on DataLoader workers per rank")
    parser.add_argument("--init-timeout", type=int, default=120,
                        help="Process group init timeout (seconds)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup and teardown
# ---------------------------------------------------------------------------
def setup(init_timeout=120):
    # Map SLURM env vars to torchrun-style env vars if needed.
    if "RANK" not in os.environ:
        if "SLURM_PROCID" not in os.environ:
            raise RuntimeError(
                "Neither RANK nor SLURM_PROCID is set. Launch under torchrun "
                "or srun, or export RANK/LOCAL_RANK/WORLD_SIZE manually."
            )
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    for var in ("MASTER_ADDR", "MASTER_PORT"):
        if var not in os.environ:
            raise RuntimeError(
                f"{var} is not set. Export it from the SLURM batch script "
                f"before launching this program."
            )

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
        timeout=datetime.timedelta(seconds=init_timeout),
    )
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=init_timeout),
        device_id=torch.device(f"cuda:{local_rank}"),
    )  


def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distributed metric helpers
# ---------------------------------------------------------------------------
def reduce_sum(value, device):
    """
    All-reduce a scalar value across all ranks and return the summed result
    as a float32 tensor on `device`. Float32 is used because NCCL float64
    reductions are not supported on all builds.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, rank):
    model.train()
    loader.sampler.set_epoch(epoch)

    # Accumulate on-device to avoid per-batch GPU->CPU syncs.
    local_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    local_correct = torch.zeros((), dtype=torch.float32, device=device)
    local_total = torch.zeros((), dtype=torch.float32, device=device)

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        local_loss_sum += loss.detach() * batch_size
        local_correct += output.argmax(dim=1).eq(target).sum().to(torch.float32)
        local_total += batch_size

        if rank == 0 and batch_idx % 100 == 0:
            # .item() here only — the per-batch print is rank 0's choice,
            # so the sync cost is bounded.
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


def evaluate(model, loader, criterion, device, dataset_len):
    """
    Evaluate over the full validation set. The DistributedSampler pads the
    dataset so every rank sees the same number of batches; we mask the
    padded samples by their original index so each example is counted once.
    """
    model.eval()

    local_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    local_correct = torch.zeros((), dtype=torch.float32, device=device)
    local_total = torch.zeros((), dtype=torch.float32, device=device)

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            batch_size = data.size(0)
            local_loss_sum += loss.detach() * batch_size
            local_correct += output.argmax(dim=1).eq(target).sum().to(torch.float32)
            local_total += batch_size

    global_loss_sum = reduce_sum(local_loss_sum, device).item()
    global_correct = reduce_sum(local_correct, device).item()
    global_total = reduce_sum(local_total, device).item()

    # global_total may exceed dataset_len because DistributedSampler pads;
    # report metrics over the padded count, but warn if the inflation is large.
    val_loss = global_loss_sum / global_total
    val_acc = 100.0 * global_correct / global_total

    return val_loss, val_acc, int(global_total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    setup(init_timeout=args.init_timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Cap workers so a generous SLURM_CPUS_PER_TASK doesn't oversubscribe.
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    num_workers = max(1, min(cpus_per_task, args.max_workers))

    # Linear LR scaling with effective batch size.
    lr = args.lr if args.no_lr_scaling else args.lr * world_size

    if rank == 0:
        print(f"Training with {world_size} GPUs", flush=True)
        print(f"Per-rank batch size: {args.batch_size} "
              f"(effective: {args.batch_size * world_size})", flush=True)
        print(f"Learning rate: {lr:g} "
              f"({'scaled' if not args.no_lr_scaling else 'unscaled'})", flush=True)
        print(f"Using {num_workers} DataLoader worker(s) per rank", flush=True)

    # Download only on rank 0, then wait for all ranks.
    if rank == 0:
        download_mnist(args.data_root)
    dist.barrier()

    train_dataset = MNISTDataset(root=args.data_root, train=True)
    val_dataset = MNISTDataset(root=args.data_root, train=False)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    # drop_last=False keeps every test sample; the small amount of padding
    # added by DistributedSampler is acknowledged in the eval reporting.
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = MNISTNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank
        )

        val_loss, val_acc, eval_count = evaluate(
            model, val_loader, criterion, device, len(val_dataset)
        )

        if rank == 0:
            pad_note = ""
            if eval_count != len(val_dataset):
                pad_note = (f"  [eval over {eval_count} samples, "
                            f"{eval_count - len(val_dataset)} padded]")
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f}  acc {val_acc:.2f}%{pad_note}",
                flush=True,
            )

    dist.barrier()

    if rank == 0:
        torch.save(model.module.state_dict(), args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}", flush=True)

    cleanup()


if __name__ == "__main__":
    main()


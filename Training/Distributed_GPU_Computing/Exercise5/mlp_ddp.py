import os
from socket import gethostname

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized

from random_dataset_gen import RandomTensorDataset


# Cap on DataLoader workers per rank. For this toy synthetic dataset,
# 0 is actually optimal; we keep a small non-zero default to exercise the
# worker path. Override by editing or via an env var if you prefer.
MAX_DATALOADER_WORKERS = int(os.environ.get("MAX_DATALOADER_WORKERS", "2"))


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_units, out_feature):
        super().__init__()
        self.hidden_layer = nn.Linear(in_feature, hidden_units)
        self.output_layer = nn.Linear(hidden_units, out_feature)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def get_local_gpu_from_rank(rank: int, gpus_per_node: int) -> int:
    """
    Map a global rank to the local GPU index on its node.
    Equivalent to rank % gpus_per_node.
    """
    return rank - gpus_per_node * (rank // gpus_per_node)


def print_rank_ordered(message: str, rank: int, world_size: int, device: int) -> None:
    """
    Print one rank at a time in ascending rank order.
    Passing device_ids=[device] to barrier avoids NCCL warnings about
    unknown device mappings.
    """
    for r in range(world_size):
        if rank == r:
            print(message, flush=True)
        dist.barrier(device_ids=[device])


def main() -> None:
    # -------------------------------------------------------------------------
    # Read distributed job information from SLURM environment variables
    # -------------------------------------------------------------------------
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])

    # Also populate RANK so any PyTorch internal that reads it (beyond the
    # explicit args to init_process_group) sees a consistent value.
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(rank % gpus_per_node))

    hostname = gethostname()
    short_hostname = hostname.split(".")[0]
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")

    visible_gpu_count = torch.cuda.device_count()
    if gpus_per_node != visible_gpu_count:
        raise RuntimeError(
            f"SLURM_GPUS_ON_NODE={gpus_per_node} but "
            f"torch.cuda.device_count()={visible_gpu_count}. "
            "Check your SLURM GPU allocation."
        )

    # -------------------------------------------------------------------------
    # Assign GPU before distributed operations that may use NCCL barriers
    # -------------------------------------------------------------------------
    device = get_local_gpu_from_rank(rank, gpus_per_node)
    torch.cuda.set_device(device)

    # -------------------------------------------------------------------------
    # Seed once, globally, so every rank builds identical initial weights.
    # (DDP also broadcasts params from rank 0 on construction, so this is
    # belt-and-suspenders; doing it here makes debugging easier and removes
    # the need to seed inside the model constructor.)
    # -------------------------------------------------------------------------
    torch.manual_seed(12345)

    # -------------------------------------------------------------------------
    # Initialize distributed process group
    # Passing device_id lets PyTorch associate this rank with its GPU, which
    # silences the "using GPU N ... devices currently unknown" warning on the
    # very first barrier.
    # -------------------------------------------------------------------------
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{device}"),
    )

    # -------------------------------------------------------------------------
    # Ordered startup print
    # -------------------------------------------------------------------------
    startup_msg = (
        f"\n"
        f"============================================================\n"
        f"Rank {rank} of {world_size}\n"
        f"Host                : {hostname}\n"
        f"Short host          : {short_hostname}\n"
        f"Local GPU assigned  : {device}\n"
        f"GPUs per node       : {gpus_per_node}\n"
        f"Visible CUDA devices: {cuda_visible_devices}\n"
        f"Process group init  : {is_initialized()}\n"
        f"============================================================"
    )
    print_rank_ordered(startup_msg, rank, world_size, device)

    # -------------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------------
    layer_1_units = 6
    layer_2_units = 4
    layer_3_units = 2

    model = MLP(
        in_feature=layer_1_units,
        hidden_units=layer_2_units,
        out_feature=layer_3_units,
    ).to(device)

    model = DDP(model, device_ids=[device])

    # -------------------------------------------------------------------------
    # Model summary
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())

    model_summary_msg = (
        f"[Rank {rank} | GPU {device}] Model Summary\n"
        f"  Input features  : {layer_1_units}\n"
        f"  Hidden units    : {layer_2_units}\n"
        f"  Output features : {layer_3_units}\n"
        f"  Total parameters: {total_params}\n"
        f"    - hidden_layer.weight: {layer_2_units * layer_1_units}\n"
        f"    - hidden_layer.bias  : {layer_2_units}\n"
        f"    - output_layer.weight: {layer_3_units * layer_2_units}\n"
        f"    - output_layer.bias  : {layer_3_units}"
    )
    print_rank_ordered(model_summary_msg, rank, world_size, device)

    # -------------------------------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------------------------------
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # -------------------------------------------------------------------------
    # Dataset and dataloader
    # -------------------------------------------------------------------------
    num_samples = 1024
    batch_size = 32

    dataset = RandomTensorDataset(
        num_samples=num_samples,
        in_shape=layer_1_units,
        out_shape=layer_3_units,
    )

    # shuffle=False here is intentional for this toy exercise; set_epoch()
    # below is a no-op in that case but is kept to illustrate best practice.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # For a tiny in-memory synthetic dataset, worker processes are pure
    # overhead. Cap at MAX_DATALOADER_WORKERS.
    num_workers = min(cpus_per_task, MAX_DATALOADER_WORKERS)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        persistent_workers=(num_workers > 0),
    )

    # Compute local batch size analytically instead of fetching a batch
    # (which would spin up workers, throw them away, then respawn for
    # the real training loop).
    samples_per_rank = (num_samples + world_size - 1) // world_size
    steps_per_rank = (samples_per_rank + batch_size - 1) // batch_size
    first_batch_size = min(batch_size, samples_per_rank)

    data_summary_msg = (
        f"[Rank {rank} | GPU {device}] Data Summary\n"
        f"  Total dataset samples : {num_samples}\n"
        f"  Local batch size      : {first_batch_size}\n"
        f"  Steps on this rank    : {steps_per_rank}\n"
        f"  World size            : {world_size}\n"
        f"  Effective global batch: {batch_size * world_size}\n"
        f"  DataLoader workers    : {num_workers}"
    )
    print_rank_ordered(data_summary_msg, rank, world_size, device)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    max_epochs = 1

    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)

        epoch_start_msg = (
            f"[Rank {rank} | GPU {device}] Starting Epoch {epoch}\n"
            f"  Steps this epoch: {len(dataloader)}"
        )
        print_rank_ordered(epoch_start_msg, rank, world_size, device)

        first_loss = None
        last_loss = None
        loss_sum = 0.0
        n_steps = 0

        for step, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if first_loss is None:
                first_loss = loss_val
            last_loss = loss_val
            loss_sum += loss_val
            n_steps += 1

        avg_loss = loss_sum / max(n_steps, 1)

        epoch_end_msg = (
            f"[Rank {rank} | GPU {device}] Finished Epoch {epoch}\n"
            f"  First local loss: {first_loss:.6e}\n"
            f"  Final local loss: {last_loss:.6e}\n"
            f"  Mean local loss : {avg_loss:.6e}"
        )
        print_rank_ordered(epoch_end_msg, rank, world_size, device)

    # -------------------------------------------------------------------------
    # Final ordered completion message
    # -------------------------------------------------------------------------
    completion_msg = (
        f"[Rank {rank} | GPU {device}] Training complete on {short_hostname}"
    )
    print_rank_ordered(completion_msg, rank, world_size, device)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    destroy_process_group()


if __name__ == "__main__":
    main()

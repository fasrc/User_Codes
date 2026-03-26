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

from random_dataset import RandomTensorDataset


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_units, out_feature):
        super().__init__()
        torch.manual_seed(12345)
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
    Using device_ids=[device] avoids NCCL barrier warnings about unknown devices.
    """
    for r in range(world_size):
        if rank == r:
            print(message, flush=True)
        dist.barrier(device_ids=[device])


def main() -> None:
    # -------------------------------------------------------------------------
    # Read distributed job information from Slurm environment variables
    # -------------------------------------------------------------------------
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])

    hostname = gethostname()
    short_hostname = os.uname().nodename.split(".")[0]
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")

    visible_gpu_count = torch.cuda.device_count()
    assert gpus_per_node == visible_gpu_count, (
        f"SLURM_GPUS_ON_NODE={gpus_per_node} vs "
        f"torch.cuda.device_count()={visible_gpu_count}"
    )

    # -------------------------------------------------------------------------
    # Assign GPU before distributed operations that may use NCCL barriers
    # -------------------------------------------------------------------------
    device = get_local_gpu_from_rank(rank, gpus_per_node)
    torch.cuda.set_device(device)

    # -------------------------------------------------------------------------
    # Initialize distributed process group
    # -------------------------------------------------------------------------
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
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

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cpus_per_task,
        sampler=sampler,
    )

    # Safe way to inspect local batch size without disrupting training logic
    first_batch = next(iter(dataloader))
    local_batch_size = len(first_batch[0])

    data_summary_msg = (
        f"[Rank {rank} | GPU {device}] Data Summary\n"
        f"  Total dataset samples : {num_samples}\n"
        f"  Local batch size      : {local_batch_size}\n"
        f"  Steps on this rank    : {len(dataloader)}\n"
        f"  World size            : {world_size}\n"
        f"  Effective global batch: {batch_size * world_size}\n"
        f"  DataLoader workers    : {cpus_per_task}"
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

        for step, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass
            out = model(x)

            # Compute loss
            loss = loss_fn(out, y)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Backward pass
            loss.backward()

            # Parameter update
            optimizer.step()

        epoch_end_msg = (
            f"[Rank {rank} | GPU {device}] Finished Epoch {epoch}\n"
            f"  Final local loss: {loss.item():.6e}"
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

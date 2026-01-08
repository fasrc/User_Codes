import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized
import os
from socket import gethostname

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

rank          = int(os.environ["SLURM_PROCID"])
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

assert (
  gpus_per_node == torch.cuda.device_count()
), f'SLURM_GPUS_ON_NODE={gpus_per_node} vs torch.cuda.device_count={torch.cuda.device_count()}'

print(
  f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
  f" {gpus_per_node} allocated GPUs per node." \
  f" | (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", flush=True
)

# Using NCCl for inter-GPU communication
init_process_group(backend="nccl", rank=rank, world_size=world_size)
if rank == 0: print(f"Group initialized? {is_initialized()}", flush=True)

device = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(device)

print(f"Using GPU{device} on Machine {os.uname().nodename.split('.')[0]} (Rank {rank})")

# model construction
layer_1_units = 6
layer_2_units = 4
layer_3_units = 2
model = MLP(
  in_feature=layer_1_units,
  hidden_units=layer_2_units,
  out_feature=layer_3_units
  ).to(device)

model = DDP(model, device_ids=[device])

# W1, b1, W2, b2 tensors are full copies (whole model replica) in each device for DDP
# Total number of parameters in each replica is equal to: 38 ==> W1(6*4) + b1(1*4) + W2(4*2) + b1(1*2)
for p in model.parameters():
  print(p)
total_params = sum(p.numel() for p in model.parameters())
print(f"[GPU{rank}] Total number of parameters in the model: {total_params}")

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

# dataset construction
num_samples = 1024
batch_size  = 32
dataset = RandomTensorDataset(
  num_samples=num_samples,
  in_shape=layer_1_units,
  out_shape=layer_3_units
  )

dataloader = DataLoader(
  dataset,
  batch_size=batch_size, # Global batch size is equal to batch_size multiply by the number of GPUs (world_size). batch_size=(batch_size//worldsize) to maintain the global batch size as batch_size
  pin_memory=True,
  shuffle=False,
  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
  sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank)
  )

max_epochs = 1
for i in range(max_epochs):
  print(f"[GPU{rank}] Epoch {i} | Batchsize: {len(next(iter(dataloader))[0])} | Steps: {len(dataloader)}")
  dataloader.sampler.set_epoch(i)
  for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
    
    # Forward Pass 
    out = model(x)

    # Calculate loss
    loss = loss_fn(out, y)

    # Zero grad
    optimizer.zero_grad(set_to_none=True)

    # Backward Pass
    loss.backward()

    # Update Model
    optimizer.step()

destroy_process_group()

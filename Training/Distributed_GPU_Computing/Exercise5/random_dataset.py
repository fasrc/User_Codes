import torch
from torch.utils.data import Dataset

class RandomTensorDataset(Dataset):
  def __init__(self, num_samples, in_shape, out_shape):
    self.num_samples = num_samples
    torch.manual_seed(12345)
    self.data = [(torch.randn(in_shape), torch.randn(out_shape)) for _ in range(num_samples)]
  
  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    return self.data[idx]
"""
Synthetic tensor dataset used by the Exercise 5 DDP example.

Every rank constructs the same dataset (same samples in the same order).
`DistributedSampler` in the training script then partitions the integer
indices [0, num_samples) into disjoint slices, one per rank, so each rank
trains on its own portion of the shared data without any cross-rank
communication of the dataset itself. That "same data on every rank"
guarantee depends on every rank using the same RNG seed to generate the
data, which is why the `seed` argument defaults to a fixed value.
"""

from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


ShapeLike = Union[int, Tuple[int, ...]]


class RandomTensorDataset(Dataset):
    """
    In-memory dataset of (input, target) pairs drawn from N(0, 1).

    Data is materialized as two contiguous tensors of shape
    (num_samples, *in_shape) and (num_samples, *out_shape). This is cheaper
    than a Python list of per-sample tuples both in memory and in the cost
    of shipping the dataset to DataLoader worker processes.

    Parameters
    ----------
    num_samples : int
        Number of (input, target) pairs to generate.
    in_shape : int or tuple of int
        Shape of each input sample (excluding the sample dimension).
    out_shape : int or tuple of int
        Shape of each target sample (excluding the sample dimension).
    seed : int, optional
        Seed for a local `torch.Generator`. Defaults to 12345 so that every
        rank that constructs the dataset with default arguments sees the
        same data. Pass `None` to draw from the ambient global RNG state
        (useful if you want non-deterministic data and are aware of the
        cross-rank consistency implications).
    """

    def __init__(
        self,
        num_samples: int,
        in_shape: ShapeLike,
        out_shape: ShapeLike,
        seed: Optional[int] = 12345,
    ) -> None:
        self.num_samples = int(num_samples)

        in_shape_t = (in_shape,) if isinstance(in_shape, int) else tuple(in_shape)
        out_shape_t = (out_shape,) if isinstance(out_shape, int) else tuple(out_shape)

        # Use a local Generator rather than mutating the global RNG state.
        # This keeps dataset construction from interfering with model
        # initialization or any other random operations in the caller.
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.x = torch.randn(self.num_samples, *in_shape_t, generator=generator)
            self.y = torch.randn(self.num_samples, *out_shape_t, generator=generator)
        else:
            self.x = torch.randn(self.num_samples, *in_shape_t)
            self.y = torch.randn(self.num_samples, *out_shape_t)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

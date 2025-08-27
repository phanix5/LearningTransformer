import numpy as np
import numpy.typing as npt
import torch


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input/target token sequences for next-token prediction.

    Args:
        x: 1D numpy array of token IDs (can be a memmap).
        batch_size: Number of sequences to sample.
        context_length: Length of each input sequence.
        device: Torch device (e.g., 'cpu', 'cuda:0', 'mps').

    Returns:
        Tuple of LongTensors (inputs, targets), each of shape (batch_size, context_length).
    """
    if x.ndim != 1:
        raise ValueError("Expected a 1D numpy array of token IDs")

    num_tokens = x.size
    if num_tokens <= context_length:
        raise ValueError("context_length must be smaller than the length of the dataset")

    max_start = num_tokens - context_length
    # Sample start indices with replacement for simplicity and good randomness
    starts = np.random.randint(0, max_start, size=batch_size)

    # Build index matrix of shape (batch_size, context_length)
    offsets = np.arange(context_length)
    input_indices = starts[:, None] + offsets[None, :]

    # Gather inputs and next-token targets from numpy
    inputs_np = x[input_indices]
    targets_np = x[input_indices + 1]

    # Move to torch tensors on the requested device with integer dtype
    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)

    return inputs, targets
    
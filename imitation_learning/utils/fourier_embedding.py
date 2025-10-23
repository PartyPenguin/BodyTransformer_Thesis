import torch
import numpy as np

def fourier_embedding(x:np.ndarray, num_harmonics, scale_factor=5 * np.pi):
    """
    Compute the Fourier embedding of a given input tensor.
    """
    x = np.expand_dims(x, axis=-1)
    harmonics = np.arange(1, num_harmonics + 1, dtype=x.dtype)
    harmonics = harmonics * scale_factor * x
    return np.concatenate([np.sin(harmonics), np.cos(harmonics)], axis=-1).reshape(
        x.shape[0], x.shape[1], -1
    )
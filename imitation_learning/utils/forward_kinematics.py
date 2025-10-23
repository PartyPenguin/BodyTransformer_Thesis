import pytorch_kinematics as pk
import torch
import numpy as np
from typing import Tuple, List, Union
from torch import Tensor


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the chain
chain = pk.build_serial_chain_from_urdf(
    open("assets/descriptions/panda_v2.urdf").read(), "panda_hand_tcp"
)


def compute_fk(q: Union[np.ndarray, Tensor], device: str = "cpu") -> Tensor:
    """
    Compute forward kinematics for each joint relative to the base frame.

    Args:
        q: Joint angles (batch_size, num_joints)
        device: Computation device

    Returns:
        Relative joint position to the base frame (3 dim) and orientation (6 dim)
        
    """
    # Convert and move to device in one step
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    chain.to(device=device)
    
    # Get all transformations at once
    all_tf = chain.forward_kinematics(q, end_only=False)
    keys = list(all_tf.keys())  # Skip base and end links
    num_links = len(keys)
    batch_size = q.shape[0]

    # Pre-allocate output tensor (batch_size, num_links-1, 9) for position and orientation
    out = torch.empty(
        (batch_size, num_links, 9), dtype=torch.float32, device=device
    )

    # Get all matrices in one go
    matrices = {k: all_tf[k].get_matrix() for k in keys}

    # Store absolute transforms from base frame
    for i in range(num_links):
        curr_tf = matrices[keys[i]]
        
        # Extract position and rotation directly (no relative transform needed)
        out[:, i, :3] = curr_tf[..., :3, 3]  # Position
        out[:, i, 3:] = curr_tf[..., :3, :2].reshape(batch_size, 6)
    
    return out


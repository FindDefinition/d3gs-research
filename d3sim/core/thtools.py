import torch 
import numpy as np 
from d3sim.constants import D3SIM_DEFAULT_DEVICE


def np_to_torch_dev(data: np.ndarray | torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = torch.device(D3SIM_DEFAULT_DEVICE)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    if data.device == device:
        return data
    return data.to(device)
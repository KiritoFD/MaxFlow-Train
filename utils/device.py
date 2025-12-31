import torch
from typing import Optional

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device.
    
    Args:
        force_cpu: If True, always use CPU regardless of GPU availability
        
    Returns:
        torch.device: The selected device
    """
    if force_cpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon support
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_info(device: torch.device) -> str:
    """Get human-readable device information."""
    if device.type == 'cuda':
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == 'mps':
        return "Apple MPS"
    else:
        return "CPU"

def get_memory_usage(device: torch.device) -> Optional[float]:
    """Get current memory usage in MB (only for CUDA)."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return None

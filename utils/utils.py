import os
import yaml
import json
from typing import Dict, Any, List
import torch

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

def ensure_dir(dir_path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

def get_device() -> torch.device:
    """Get GPU device if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, checkpoint_path: str) -> None:
    """Save model checkpoint"""
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

from torch.utils.data import Dataset
from typing import Dict
import torch
from src.data_module.dataset import register_dataset


@register_dataset("simple")
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.data[i]
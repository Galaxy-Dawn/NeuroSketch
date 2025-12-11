import torch
import numpy as np


class Mixup:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_imgs: torch.Tensor = lam * imgs + (1 - lam) * imgs[idx]
        mixed_labels: torch.Tensor = lam * labels + (1 - lam) * labels[idx]
        return mixed_imgs, mixed_labels
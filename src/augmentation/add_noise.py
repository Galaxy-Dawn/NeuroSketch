import torch


class add_noise:
    def __init__(self, snrs=(15, 30), dims=(1, 2)):
        self.snrs = snrs
        self.dims = dims

    def __call__(self, imgs):
        if isinstance(self.snrs, (list, tuple)):
            snr = (self.snrs[0] - self.snrs[1]) * torch.rand((imgs.shape[0],), device=imgs.device).reshape(-1, 1, 1) + self.snrs[1]
        else:
            snr = self.snrs
        snr = 10 ** (snr / 20)
        sigma = torch.std(imgs, dim=self.dims, keepdim=True) / snr
        return imgs + torch.randn(imgs.shape, device=imgs.device) * sigma
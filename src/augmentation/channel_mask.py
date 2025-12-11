import torch


class ChannelMasking:
    def __init__(self, mask_prob=0.2):
        self.mask_prob = mask_prob

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        对输入的批量图像应用通道掩码。
        Args:
            imgs (torch.Tensor): 输入张量，形状为 [batch, channel, timestep]
        Returns:
            torch.Tensor: 掩码后的张量
        """
        batch_size, num_channels, _ = imgs.size()
        mask = torch.rand(batch_size, num_channels, 1, device=imgs.device) < self.mask_prob
        imgs = imgs.clone()
        imgs[mask.expand_as(imgs)] = 0
        return imgs
import torch


class TimeMasking:
    def __init__(self, mask_param=50, num_masks=4):
        """
        初始化时间掩码。
        Args:
            mask_param (int): 每个掩码的最大长度。
            num_masks (int): 每个样本应用的掩码数量。
        """
        self.mask_param = mask_param
        self.num_masks = num_masks

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        对输入的批量图像应用时间掩码。
        Args:
            imgs (torch.Tensor): 输入张量，形状为 [batch, channel, timestep]
        Returns:
            torch.Tensor: 掩码后的张量
        """
        batch_size, _, timestep = imgs.size()
        imgs = imgs.clone()
        for i in range(batch_size):
            for _ in range(self.num_masks):
                mask_length = torch.randint(0, self.mask_param, (1,)).item()
                if mask_length == 0:
                    continue
                start = torch.randint(0, max(timestep - mask_length, 1), (1,)).item()
                imgs[i, :, start:start + mask_length] = 0
        return imgs
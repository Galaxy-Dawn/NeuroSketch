import torch
import torch.nn as nn
from omegaconf import ListConfig

from src.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup
import random
from einops import rearrange
import torch.nn.functional as F
from src.model_module.model import register_model


class PatchEmbedding1DConv(nn.Module):
    def __init__(self, input_size, emb_size1, emb_size2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(input_size, emb_size1, 5, 1, padding=2),
            nn.BatchNorm1d(emb_size1),
            nn.ReLU(),
            nn.Conv1d(emb_size1, emb_size2, 5, 1, padding=2),
        )
    def forward(self, x):
        x = self.projection(x)
        return x

class ResNet_1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class ChannelCNN(nn.Module):
    def __init__(self, input_size, kernel_size, blocks):
        super().__init__()
        self.resnet = self._make_resnet_layer(input_size=input_size, kernel_size=kernel_size, blocks=blocks)
    def _make_resnet_layer(self, input_size, kernel_size, blocks=9):
        layers = []
        for i in range(blocks):
            layers.append(ResNet_1D_Block(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size,
                                          stride=1, padding=(kernel_size-1)//2))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        # self.bn1 = nn.BatchNorm1d(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)
        res = self.fc1(res)
        res = self.ln1(res)
        # res = res.permute(0, 2, 1)
        # res = self.bn1(res)
        # res = res.permute(0, 2, 1)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        # res = res.permute(0, 2, 1)
        # res = self.bn2(res)
        # res = res.permute(0, 2, 1)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h


@register_model('GRU')
class MultiResidualBiGRU(nn.Module):
    def __init__(self, cfg):
        super(MultiResidualBiGRU, self).__init__()
        self.input_size = cfg.dataset.input_channels if not isinstance(cfg.dataset.input_channels, ListConfig) else cfg.dataset.input_channels[cfg.dataset.id]
        self.hidden_size = cfg.model.hidden_size
        self.n_layers = cfg.model.n_layers
        self.bidir = cfg.model.bidir
        self.cfg = cfg
        self.use_channel_cnn = cfg.model.use_channel_cnn
        self.use_patch = cfg.model.use_patch
        self.patch_len = cfg.model.patch_len
        self.patch_stride = cfg.model.patch_stride
        self.patch_embedding = PatchEmbedding1DConv(self.input_size * self.patch_len, self.hidden_size // 2,
                                                     self.hidden_size)
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)
        # self.bn = nn.BatchNorm1d(self.hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(self.hidden_size, n_layers=1, bidir=self.bidir)
                for _ in range(self.n_layers)
            ]
        )
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
        self.fc_out = nn.Linear(self.hidden_size, self.target_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def apply_augmentation(self, x):
        augmentations = [
            (random_shift, self.cfg.augmentation.random_shift_prob),
            (add_noise(), self.cfg.augmentation.add_noise_prob),
            (ChannelMasking(), self.cfg.augmentation.ChannelMasking_prob),
            (TimeMasking(), self.cfg.augmentation.TimeMasking_prob),
        ]
        for aug_func, prob in augmentations:
            if random.random() < prob:
                x = aug_func(x)
        return x

    def patchify(self, x, patch_len, patch_stride):
        """
        将输入张量 x 进行分 patch 处理。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, channels, seq_len)
        patch_len (int): 每个 patch 的长度
        patch_stride (int): 每个 patch 的步长
        返回:
        torch.Tensor: 分 patch 后的张量，形状为 (batch_size, channels, num_patches, patch_len)
        """
        # num_patches = (seq_len - patch_len) // patch_stride + 1
        x = x.unfold(dimension=-1, size=patch_len, step=patch_stride)
        return x

    def forward(self, ieeg_raw_data, labels, **kwargs):
        x = ieeg_raw_data
        if type(labels) != torch.LongTensor:
            labels = labels.long()
        one_hot_labels = F.one_hot(labels, num_classes=self.target_size).float()
        if self.training:
            x = self.apply_augmentation(x)
            p = random.random()
            if p < self.cfg.augmentation.mixup_prob:
                x, one_hot_labels = Mixup(alpha=0.5)(x, one_hot_labels)
        x = x.float()

        if self.use_channel_cnn or self.use_patch:
            if self.use_channel_cnn:
                x = self.channel_cnn(x)
            if self.use_patch:
                x = self.patchify(x, self.cfg.model.patch_len, self.cfg.model.patch_stride)
                x = rearrange(x, 'b c s p -> b s (c p)')
                x = rearrange(x, 'b s p -> b p s')
                # b (c p) s
                x = self.patch_embedding(x)
                # b d s
                x = rearrange(x, 'b d s -> b s d')
                # b s d
        else:
            x = x.permute(0, 2, 1)
            x = self.fc_in(x)
        h = [None for _ in range(self.n_layers)]
        x = self.ln(x)
        x = nn.functional.relu(x)
        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
        prediction = self.fc_out(x.mean(1))
        loss1 = self.loss_fn(prediction, one_hot_labels)
        # loss = (loss1 - self.b).abs() + self.b
        loss = loss1
        return {"loss"  : loss,
                "labels": labels,
                "logits": prediction}







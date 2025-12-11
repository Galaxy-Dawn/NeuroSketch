from functools import partial
import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
import torch.nn.functional as F
from src.model_module.utils import GeM
from omegaconf import DictConfig, ListConfig
import random
from src.model_module.model import register_model
from src.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup

NORM_EPS = 1e-5

class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size-1)//2,
                              groups=groups,
                              bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm1d, eps=NORM_EPS)
        if stride == 2:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(x))


class CB(nn.Module):
    def __init__(self, out_channels):
        super(CB, self).__init__()
        norm_layer = partial(nn.BatchNorm1d, eps=NORM_EPS)
        self.conv3x1 = nn.Conv1d(out_channels, out_channels,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()
        self.projection = nn.Conv1d(out_channels, out_channels,
                                    kernel_size=1,
                                    bias=False)

    def forward(self, x):
        out = self.conv3x1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class NCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0, drop=0):
        super(NCB, self).__init__()
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.cb = CB(out_channels)
        self.path_dropout = DropPath(path_dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.path_dropout(self.cb(x))
        return x


class CNN_backbone(nn.Module):
    def __init__(self, cfg, stem_chs, depths, path_dropout, drop=0, strides=[1, 2, 2, 2], use_checkpoint=False):
        super(CNN_backbone, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = cfg.dataset.input_channels if not isinstance(cfg.dataset.input_channels, ListConfig) else cfg.dataset.input_channels[cfg.dataset.id]
        self.stage_out_channels = [
            [cfg.model.stage1_dim] * depths[0],
            [cfg.model.stage2_dim] * depths[1],
            [cfg.model.stage3_dim] * depths[2],
            [cfg.model.embed_dim] * depths[3]
        ]

        self.stem = nn.Sequential(
            ConvBNReLU(self.in_channels, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], cfg.model.stage1_dim, kernel_size=3, stride=2),
        )

        input_channel = cfg.model.stage1_dim
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]

        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            for block_id in range(numrepeat):
                stride = strides[stage_id] if block_id == 0 else 1
                output_channel = output_channels[block_id]
                layer = NCB(input_channel, output_channel,
                            stride=stride,
                            path_dropout=dpr[idx + block_id],
                            drop=drop)
                features.append(layer)
                input_channel = output_channel
            idx += numrepeat

        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm1d(output_channel, eps=NORM_EPS)
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)


@register_model('CNN_1D')
class CNN_1D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(CNN_1D, self).__init__()
        self.cfg = cfg
        self.encoder = CNN_backbone(
            cfg,
            stem_chs=[64, 256, 64],
            depths=[
                cfg.model.stage1_depth,
                cfg.model.stage2_depth,
                cfg.model.stage3_depth,
                cfg.model.stage4_depth
            ],
            path_dropout=0.2
        )
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
        self.pool = GeM(p_trainable=True, module=1)
        self.fc = nn.Linear(cfg.model.embed_dim, self.target_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)
        logits = self.fc(pooled)
        loss = self.loss_fn(logits, one_hot_labels)
        return {
            "loss"  : loss,
            "labels": labels,
            "logits": logits
        }
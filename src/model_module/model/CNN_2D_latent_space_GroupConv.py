from functools import partial
import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
import torch.nn.functional as F
from src.model_module.utils import GeM
from omegaconf import DictConfig
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size-1)//2, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
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
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()
    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class CB(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, kernel_size, out_channels, head_dim):
        super(CB, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                                       padding=(kernel_size-1)//2, groups=head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class NCB(nn.Module):
    def __init__(self, seg_dim, in_channels, out_channels, head_dim, stride=1, path_dropout=0, residual=True):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.patch_embed = PatchEmbed(seg_dim, in_channels, out_channels, stride)
        self.cb = CB(seg_dim, out_channels, head_dim=head_dim)
        self.attention_path_dropout = DropPath(path_dropout)
        self.residual = residual
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.residual:
            x = x + self.attention_path_dropout(self.cb(x))
        else:
            x = self.attention_path_dropout(self.cb(x))
        return x


class CNN_backbone(nn.Module):
    def __init__(self, cfg, stem_chs, depths, path_dropout, drop=0, use_checkpoint=False):
        super(CNN_backbone, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.stage_out_channels = [[cfg.model.stage1_dim] * (depths[0]),
                                   [cfg.model.stage2_dim] * (depths[1]),
                                   [cfg.model.stage3_dim] * (depths[2]),
                                   [cfg.model.embed_dim] * (depths[3])]
        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * depths[1],
                                  [NCB] * depths[2],
                                  [NCB] * depths[3]]
        self.stem = nn.Sequential(
            ConvBNReLU(1, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], cfg.model.stage1_dim, kernel_size=3, stride=2),
        )
        input_channel = cfg.model.stage1_dim
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            for block_id in range(numrepeat):
                if stage_id == 1 and block_id <= 2:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                layer = NCB(cfg.model.seg_dim,
                            input_channel,
                            output_channel,
                            head_dim = cfg.model.head_dim,
                            stride=stride,
                            path_dropout=dpr[idx + block_id],
                            )
                features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        print('initialize_weights...')
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


@register_model('CNN_2D_latent_space_GroupConv')
class CNN_2D_latent_space_GroupConv(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(CNN_2D_latent_space_GroupConv, self).__init__()
        self.cfg = cfg
        self.encoder = CNN_backbone(cfg, stem_chs=[64, 32, 64], depths=[cfg.model.stage1_depth, cfg.model.stage2_depth, cfg.model.stage3_depth, cfg.model.stage4_depth], path_dropout=0.2)
        self.seg_dim = cfg.model.seg_dim
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
        self.pool = GeM(p_trainable=True)
        self.fc = nn.Linear(cfg.model.embed_dim, out_features=self.target_size, bias=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        feature1 = self.encoder.forward_features(x)
        return feature1

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

    def adjust_length_and_reshape(self, seg_dim, x):
        bs, ch, l = x.size()
        remainder = l % seg_dim
        if remainder != 0:
            l_new = l + (seg_dim - remainder)  # 向上取最近的倍数
        else:
            l_new = l
        if l_new != l:
            x = F.pad(x, (0, l_new - l))  # 填充
        x = x.view(bs, ch, l_new // seg_dim, seg_dim)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(bs, ch * seg_dim, l_new // seg_dim)
        x = torch.unsqueeze(x, dim=1)
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
        bs, ch, l = x.size(0), x.size(1), x.size(2)
        x = self.adjust_length_and_reshape(self.seg_dim, x)
        # x = torch.cat([x, x, x], dim=1)
        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        prediction = self.fc(x)
        loss1 = self.loss_fn(prediction, one_hot_labels)
        loss = loss1
        return {"loss": loss,
                "labels": labels,
                "logits": prediction}
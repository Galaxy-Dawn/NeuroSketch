import torch
import torch.nn.functional as F
from torch import nn
from src.model_module.utils import GeM
from omegaconf import DictConfig
import random
from src.model_module.model import register_model
from src.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from src.utils.get_act import get_act

NORM_EPS = 1e-5

def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1,
            act_type="ReLU"
    ):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size-1)//2, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = get_act(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            # self.avgpool = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim, act_type="ReLU", cnn_dropout=0):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = get_act(act_type)
        self.dropout = nn.Dropout(p=cnn_dropout)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True, act_type="ReLU"):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = get_act(act_type)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0.0, act_type="ReLU",
                 cnn_dropout=0.0, mlp_dropout=0.0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim, act_type, cnn_dropout)
        self.cnn_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)

        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=mlp_dropout, bias=True, act_type=act_type)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.cnn_path_dropout(self.mhca(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class NeuralMonsterEncoder(nn.Module):
    def __init__(self, stem_chs, depths, act_type, path_dropout, cnn_dropout=0.0, mlp_dropout=0.0,
                 strides=[1, 2, 2, 2], head_dim=32, mlp_ratio=3,
                 use_checkpoint=False):
        super(NeuralMonsterEncoder, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [[96] * (depths[0] - 1) + [128],
                                   [192] * (depths[1] - 1) + [256],
                                   [384] * (depths[2] - 1) + [512],
                                   [768] * (depths[3] - 1) + [1024]]

        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1]),
                                  [NCB] * (depths[2]),
                                  [NCB] * (depths[3])]

        self.stem = nn.Sequential(
            ConvBNAct(1, stem_chs[0], kernel_size=3, stride=2, act_type=act_type),
            ConvBNAct(stem_chs[0], stem_chs[1], kernel_size=3, stride=1, act_type=act_type),
            ConvBNAct(stem_chs[1], stem_chs[2], kernel_size=3, stride=1, act_type=act_type),
            ConvBNAct(stem_chs[2], stem_chs[2], kernel_size=3, stride=2, act_type=act_type),
        )

        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel,
                                output_channel,
                                stride=stride,
                                path_dropout=dpr[idx + block_id],
                                cnn_dropout=cnn_dropout,
                                mlp_dropout=mlp_dropout,
                                head_dim=head_dim,
                                mlp_ratio=mlp_ratio)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, NCB):
                module.merge_bn()

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


@register_model('NeuroSketch_sd')
class NeuroSketch_sd(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(NeuroSketch_sd, self).__init__()
        self.cfg = cfg
        self.encoder = NeuralMonsterEncoder(stem_chs=[64, 32, 64],
                                 depths=[cfg.model.stage1_depth, cfg.model.stage2_depth, cfg.model.stage3_depth, cfg.model.stage4_depth],
                                 act_type=cfg.model.act_type,
                                 path_dropout=cfg.model.drop_path_rate,
                                 cnn_dropout=cfg.model.cnn_dropout,
                                 mlp_dropout=cfg.model.mlp_dropout,
                                 head_dim=cfg.model.head_dim,
                                 mlp_ratio=cfg.model.mlp_ratio)
        self.seg_dim = cfg.model.seg_dim
        self.target_size = cfg.dataset.target_size if not isinstance(cfg.dataset.target_size, DictConfig) else \
            cfg.dataset.target_size[cfg.dataset.task]
        self.pool = GeM(p_trainable=True)
        self.fc = nn.Linear(cfg.encoder.embed_dim, out_features=self.target_size, bias=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        feature = self.encoder.forward_features(x)
        return feature

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
        if seg_dim != 1:
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
        else:
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
        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        prediction = self.fc(x)
        loss1 = self.loss_fn(prediction, one_hot_labels)
        loss = loss1
        return {"loss": loss,
                "labels": labels,
                "logits": prediction}
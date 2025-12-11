import torch
import torch.nn as nn
from omegaconf import ListConfig
from src.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup
import random
import torch.nn.functional as F
from src.model_module.model.CNN_1D import CNN_backbone
from src.model_module.model.GRU import ResidualBiGRU
from src.model_module.model import register_model


@register_model('CNN_GRU')
class CNN_GRU(nn.Module):
    def __init__(self, cfg):
        super(CNN_GRU, self).__init__()
        # CNN特征提取部分
        self.cnn_backbone = CNN_backbone(
            cfg,
            stem_chs=[64, 32, 64],
            depths=[
                cfg.model.stage1_depth,
                cfg.model.stage2_depth,
                cfg.model.stage3_depth,
                cfg.model.stage4_depth
            ],
            path_dropout=0.2
        )

        # GRU时序建模部分
        self.gru_hidden_size = cfg.model.gru_hidden_size
        self.n_layers = cfg.model.gru_layers
        self.bidir = cfg.model.bidir

        # 维度适配层
        self.cnn_feat_dim = cfg.model.embed_dim  # CNN最终输出维度
        self.fc_adapt = nn.Linear(self.cnn_feat_dim, self.gru_hidden_size)

        # 残差GRU模块
        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(self.gru_hidden_size, n_layers=1, bidir=self.bidir)
            for _ in range(self.n_layers)
        ])

        # 输出层
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
        self.fc_out = nn.Linear(self.gru_hidden_size, self.target_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # 其他参数
        self.cfg = cfg
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def apply_augmentation(self, x):
        # 保持与原有一致的增强逻辑
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
        # 标签处理
        if type(labels) != torch.LongTensor:
            labels = labels.long()
        one_hot_labels = F.one_hot(labels, num_classes=self.target_size).float()
        # 数据增强
        if self.training:
            x = self.apply_augmentation(x)
            p = random.random()
            if p < self.cfg.augmentation.mixup_prob:
                x, one_hot_labels = Mixup(alpha=0.5)(x, one_hot_labels)
        # CNN特征提取 [batch, channels, seq_len]
        cnn_features = self.cnn_backbone(x.float())
        # 维度调整 [batch, channels, seq_len] -> [batch, seq_len, features]
        x = cnn_features.permute(0, 2, 1)  # [B, Seq_len, CNN_feat]
        # 维度适配
        x = self.fc_adapt(x)  # [B, Seq_len, gru_hidden]

        # GRU时序建模
        h = [None for _ in range(self.n_layers)]
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])

        # 输出处理
        prediction = self.fc_out(x.mean(1))  # 沿时间维度平均
        loss = self.loss_fn(prediction, one_hot_labels)

        return {
            "loss"  : loss,
            "labels": labels,
            "logits": prediction
        }

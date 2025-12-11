import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from omegaconf import ListConfig
from torch import Tensor
from src.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup
import random
import math
from src.model_module.model import register_model


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding1DConvWithDownsample(nn.Module):
    def __init__(self, input_size, emb_size1, emb_size2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(input_size, emb_size1, 21, 10, padding=10),
            nn.BatchNorm1d(emb_size1),
            nn.ReLU(),
            nn.Conv1d(emb_size1, emb_size2, 13, 6, padding=6),
        )
    def forward(self, x):
        x = self.projection(x)
        return x


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


class PatchEmbeddingInverted(nn.Module):
    def __init__(self, input_size, emb_size1, emb_size2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(input_size, emb_size1, kernel_size=1),
            nn.BatchNorm1d(emb_size1),
            nn.ReLU(),
            nn.Conv1d(emb_size1, emb_size2, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, sequence_num=2000, inter=100, n_channels=16):
        super(ChannelAttention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(
                n_channels
            ),  # also may introduce improvement to a certain extent
            nn.Dropout(0.0),
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.0),
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.0),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b c s->b s c")
        temp_query = rearrange(self.query(temp), "b s c -> b c s")
        temp_key = rearrange(self.key(temp), "b s c -> b c s")

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b c s, b m s -> b c m", channel_query, channel_key) / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b c s, b c m -> b c s", x, channel_atten_score)
        """
        projections after or before multiplying with attention score are almost the same.
        """
        out = rearrange(out, "b c s -> b s c")
        out = self.projection(out)
        out = rearrange(out, "b s c -> b c s")
        return out


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

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, ff_hidden_size=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention block with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))
        # Feed-forward block with residual connection
        ff_out = self.ff(x)
        x = self.layer_norm2(x + self.dropout(ff_out))
        return x


@register_model('Transformer')
class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.input_size = cfg.dataset.input_channels if not isinstance(cfg.dataset.input_channels, ListConfig) else cfg.dataset.input_channels[cfg.dataset.id]
        self.hidden_size = cfg.model.hidden_size
        self.out_size = cfg.dataset.target_size[cfg.dataset.task]
        self.n_layers = cfg.model.n_layers
        self.num_heads = cfg.model.num_heads
        self.ff_hidden_size = cfg.model.ff_hidden_size
        self.input_type = cfg.model.input_type
        self.patch_len = cfg.model.patch_len
        self.patch_stride = cfg.model.patch_stride
        self.unpatch_seq_len = cfg.dataset.seq_len[cfg.dataset.task]
        self.seq_len = (cfg.dataset.seq_len[cfg.dataset.task] - self.patch_len)// self.patch_stride + 1
        self.cfg = cfg
        self.use_channel_attention = cfg.model.use_channel_attention
        self.use_channel_cnn = cfg.model.use_channel_cnn
        self.channel_cnn = ChannelCNN(input_size=self.input_size, kernel_size=5, blocks=5)
        self.channel_attention = ChannelAttention(sequence_num=cfg.dataset.seq_len[cfg.dataset.task], inter=5, n_channels=self.input_size)
        self.patch_embedding1 = PatchEmbedding1DConvWithDownsample(self.input_size, self.hidden_size//2, self.hidden_size)
        self.patch_embedding2 = PatchEmbedding1DConv(self.input_size * self.patch_len, self.hidden_size//2, self.hidden_size)
        self.patch_embedding3 = PatchEmbedding1DConv(self.patch_len, self.hidden_size//2, self.hidden_size)
        self.patch_embedding4 = PatchEmbeddingInverted(self.unpatch_seq_len, self.hidden_size//2, self.hidden_size)
        self.position_embedding = PositionalEmbedding(self.hidden_size)
        # List of Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden_size,
                                         num_heads=self.num_heads,
                                         ff_hidden_size=self.ff_hidden_size)
                for _ in range(self.n_layers)
            ]
        )
        self.fc_out1 = nn.Linear(self.hidden_size * self.input_size, self.out_size)
        self.fc_out2 = nn.Linear(self.hidden_size, self.out_size)
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
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
        if self.use_channel_attention:
            x = self.channel_attention(x)
        elif self.use_channel_cnn:
            x = self.channel_cnn(x)

        if self.input_type == 1:  #No Patchify Ori Transformer
            # b c l
            x = self.patch_embedding1(x)
            # b d s
            x = rearrange(x, 'b d s -> b s d')
            #b s d
        elif self.input_type == 4:
            x = self.patch_embedding4(x)
        else:
            x = self.patchify(x, self.cfg.model.patch_len, self.cfg.model.patch_stride)
            bsz, ch_num, seq_len, patch_len = x.shape
            if self.input_type == 2:  #Our Patchify Transformer
                x = rearrange(x, 'b c s p -> b s (c p)')
                x = rearrange(x, 'b s p -> b p s')
                # b (c p) s
                x = self.patch_embedding2(x)
                # b d s
                x = rearrange(x, 'b d s -> b s d')
                # b s d
            elif self.input_type == 3: #Channel Independent PatchTST
                x = rearrange(x, 'b c s p -> (b c) s p')
                x = rearrange(x, 'b s p -> b p s')
                # (b c) p s
                x = self.patch_embedding3(x)
                # (b c) d s
                x = rearrange(x, 'b d s -> b s d')
                # (b c) s d
            else:
                raise ValueError("Invalid patch_type value")

        x = x + self.position_embedding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        if self.input_type == 3:
            x = rearrange(x, '(b c) s d -> b c s d', b = bsz, c = ch_num, s = seq_len)
            x = reduce(x, 'b c s d -> b c d', 'mean')
            x = rearrange(x, 'b c d -> b (c d)')
            prediction = self.fc_out1(x)
        else:
            x = reduce(x, 'b s d -> b d', 'mean')
            prediction = self.fc_out2(x)

        # Calculate loss
        loss = self.loss_fn(prediction, one_hot_labels)
        return {
            "loss"  : loss,
            "labels": labels,
            "logits": prediction
        }

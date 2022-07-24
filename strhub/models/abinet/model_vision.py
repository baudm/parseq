from torch import nn

from .attention import PositionAttention, Attention
from .backbone import ResTranformer
from .model import Model
from .resnet import resnet45


class BaseVision(Model):
    def __init__(self, dataset_max_length, null_label, num_classes,
                 attention='position', attention_mode='nearest', loss_weight=1.0,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 backbone='transformer', backbone_ln=2):
        super().__init__(dataset_max_length, null_label)
        self.loss_weight = loss_weight
        self.out_channels = d_model

        if backbone == 'transformer':
            self.backbone = ResTranformer(d_model, nhead, d_inner, dropout, activation, backbone_ln)
        else:
            self.backbone = resnet45()

        if attention == 'position':
            self.attention = PositionAttention(
                max_length=self.max_length,
                mode=attention_mode
            )
        elif attention == 'attention':
            self.attention = Attention(
                max_length=self.max_length,
                n_feature=8 * 32,
            )
        else:
            raise ValueError(f'invalid attention: {attention}')

        self.cls = nn.Linear(self.out_channels, num_classes)

    def forward(self, images):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision'}

import torch
import torch.nn as nn

from .model import Model


class BaseAlignment(Model):
    def __init__(self, dataset_max_length, null_label, num_classes, d_model=512, loss_weight=1.0):
        super().__init__(dataset_max_length, null_label)
        self.loss_weight = loss_weight
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight': self.loss_weight,
                'name': 'alignment'}

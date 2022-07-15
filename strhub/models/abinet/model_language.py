import torch.nn as nn
from torch.nn import TransformerDecoder

from .model import Model
from .transformer import PositionalEncoding, TransformerDecoderLayer


class BCNLanguage(Model):
    def __init__(self, dataset_max_length, null_label, num_classes, d_model=512, nhead=8, d_inner=2048, dropout=0.1,
                 activation='relu', num_layers=4, detach=True, use_self_attn=False, loss_weight=1.0,
                 global_debug=False):
        super().__init__(dataset_max_length, null_label)
        self.detach = detach
        self.loss_weight = loss_weight
        self.proj = nn.Linear(num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                                                activation, self_attn=use_self_attn, debug=global_debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                            tgt_key_padding_mask=padding_mask,
                            memory_mask=location_mask,
                            memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'language'}
        return res

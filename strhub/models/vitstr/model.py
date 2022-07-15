'''
Implementation of ViTSTR based on timm VisionTransformer.

TODO: 
1) distilled deit backbone
2) base deit backbone

Copyright 2021 Rowel Atienza
'''

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer


class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen=25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x

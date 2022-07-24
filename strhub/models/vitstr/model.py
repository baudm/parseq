"""
Implementation of ViTSTR based on timm VisionTransformer.

TODO:
1) distilled deit backbone
2) base deit backbone

Copyright 2021 Rowel Atienza
"""

from timm.models.vision_transformer import VisionTransformer


class ViTSTR(VisionTransformer):
    """
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    """

    def forward(self, x, seqlen: int = 25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x

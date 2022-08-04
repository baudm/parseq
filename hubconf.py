from strhub.models.utils import create_model


dependencies = ['torch', 'pytorch_lightning', 'timm']


def parseq_tiny(pretrained: bool = False, decode_ar: bool = True, refine_iters: int = 1, **kwargs):
    """
    PARSeq tiny model (img_size=128x32, patch_size=8x4, d_model=192)
    @param pretrained: (bool) Use pretrained weights
    @param decode_ar: (bool) use AR decoding
    @param refine_iters: (int) number of refinement iterations to use
    """
    return create_model('parseq-tiny', pretrained, decode_ar=decode_ar, refine_iters=refine_iters, **kwargs)


def parseq(pretrained: bool = False, decode_ar: bool = True, refine_iters: int = 1, **kwargs):
    """
    PARSeq base model (img_size=128x32, patch_size=8x4, d_model=384)
    @param pretrained: (bool) Use pretrained weights
    @param decode_ar: (bool) use AR decoding
    @param refine_iters: (int) number of refinement iterations to use
    """
    return create_model('parseq', pretrained, decode_ar=decode_ar, refine_iters=refine_iters, **kwargs)


def abinet(pretrained: bool = False, iter_size: int = 3, **kwargs):
    """
    ABINet model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    @param iter_size: (int) number of refinement iterations to use
    """
    return create_model('abinet', pretrained, iter_size=iter_size, **kwargs)


def trba(pretrained: bool = False, **kwargs):
    """
    TRBA model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    """
    return create_model('trba', pretrained, **kwargs)


def vitstr(pretrained: bool = False, **kwargs):
    """
    ViTSTR small model (img_size=128x32, patch_size=8x4, d_model=384)
    @param pretrained: (bool) Use pretrained weights
    """
    return create_model('vitstr', pretrained, **kwargs)


def crnn(pretrained: bool = False, **kwargs):
    """
    CRNN model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    """
    return create_model('crnn', pretrained, **kwargs)

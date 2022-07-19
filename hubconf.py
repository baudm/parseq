from pathlib import PurePath

import torch
import yaml


dependencies = ['torch', 'pytorch_lightning', 'timm']


def _get_config(model: str, experiment: str = None, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parent
    with open(root.joinpath('configs/main.yaml'), 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root.joinpath(f'configs/charset/94_full.yaml'), 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root.joinpath(f'configs/model/{model}.yaml'), 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    if experiment is not None:
        with open(root.joinpath(f'configs/experiment/{experiment}.yaml'), 'r') as f:
            config.update(yaml.load(f, yaml.Loader)['model'])
    config.update(kwargs)
    return config


def parseq_tiny(pretrained: bool = False, decode_ar: bool = True, refine_iters: int = 1, **kwargs):
    """
    PARSeq tiny model (img_size=128x32, patch_size=8x4, d_model=192)
    @param pretrained: (bool) Use pretrained weights
    @param decode_ar: (bool) use AR decoding
    @param refine_iters: (int) number of refinement iterations to use
    """
    from strhub.models.parseq.system import PARSeq
    config = _get_config('parseq', 'parseq-tiny', decode_ar=decode_ar, refine_iters=refine_iters, **kwargs)
    model = PARSeq(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


def parseq(pretrained: bool = False, decode_ar: bool = True, refine_iters: int = 1, **kwargs):
    """
    PARSeq base model (img_size=128x32, patch_size=8x4, d_model=384)
    @param pretrained: (bool) Use pretrained weights
    @param decode_ar: (bool) use AR decoding
    @param refine_iters: (int) number of refinement iterations to use
    """
    from strhub.models.parseq.system import PARSeq
    config = _get_config('parseq', decode_ar=decode_ar, refine_iters=refine_iters, **kwargs)
    model = PARSeq(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


def abinet(pretrained: bool = False, iter_size: int = 3, **kwargs):
    """
    ABINet model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    @param iter_size: (int) number of refinement iterations to use
    """
    from strhub.models.abinet.system import ABINet
    config = _get_config('abinet', iter_size=iter_size, **kwargs)
    model = ABINet(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


def trba(pretrained: bool = False, **kwargs):
    """
    TRBA model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    """
    from strhub.models.trba.system import TRBA
    config = _get_config('trba', **kwargs)
    model = TRBA(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


def vitstr(pretrained: bool = False, **kwargs):
    """
    ViTSTR small model (img_size=128x32, patch_size=8x4, d_model=384)
    @param pretrained: (bool) Use pretrained weights
    """
    from strhub.models.vitstr.system import ViTSTR
    config = _get_config('vitstr', 'vitstr', **kwargs)
    model = ViTSTR(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


def crnn(pretrained: bool = False, **kwargs):
    """
    CRNN model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    """
    from strhub.models.crnn.system import CRNN
    config = _get_config('crnn', **kwargs)
    model = CRNN(**config)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
            map_location='cpu', check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

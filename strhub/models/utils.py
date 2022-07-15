import os.path
from typing import Sequence

import torch
from pytorch_lightning.core.saving import ModelIO
from torch import nn


def _load_pl_checkpoint(checkpoint, **kwargs):
    hparams = checkpoint[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY]
    hparams.update(kwargs)
    name = hparams['name']
    if name.startswith('abinet'):
        from .abinet.system import ABINet as ModelClass
    elif name.startswith('crnn'):
        from .crnn.system import CRNN as ModelClass
    elif name.startswith('parseq'):
        from .parseq.system import PARSeq as ModelClass
    elif name.startswith('trba'):
        from .trba.system import TRBA as ModelClass
    elif name.startswith('trbc'):
        from .trba.system import TRBC as ModelClass
    elif name.startswith('vitstr'):
        from .vitstr.system import ViTSTR as ModelClass
    else:
        raise RuntimeError('Unable to load correct model class')
    model = ModelClass._load_model_state(checkpoint, strict=True, **kwargs)
    return model


def _load_torch_model(checkpoint_path, checkpoint, **kwargs):
    import hubconf
    name = os.path.basename(checkpoint_path).split('-')[0]
    model_factory = getattr(hubconf, name)
    model = model_factory(**kwargs)
    model.load_state_dict(checkpoint)
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        model = _load_pl_checkpoint(checkpoint, **kwargs)
    except KeyError:
        model = _load_torch_model(checkpoint_path, checkpoint, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

import os
from typing import Any, Callable, Dict, Iterable, List
from typing import Tuple

import torch

from dprt.models.dprt import build_dprt


def build(model: str, *args, **kwargs):
    if model == 'dprt':
        return build_dprt(*args, **kwargs)


def load(checkpoint: str, config: Dict[str, Any] = None, *args, **kwargs) -> Tuple[torch.nn.Module, int, str]:
    filename = os.path.splitext(os.path.basename(checkpoint))[0]
    timestamp, _, epoch = filename.split('_')
    if config is not None:
        model = build(config["model"]["name"], config)
        model.load_state_dict(torch.load(checkpoint)),
        return model, int(epoch), timestamp
    return torch.load(checkpoint), int(epoch), timestamp

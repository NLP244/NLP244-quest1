import torch
import numpy as np
import random
import os


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # for new Mac M1 or M2 chips
    else:
        device = torch.device("cpu")
    return device


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_reproducible(seed:int = 42) -> None:
    """
    Set random seed in a bunch of libraries.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _numpy_rng = np.random.default_rng(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

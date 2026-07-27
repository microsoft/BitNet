# -*- coding: utf-8 -*-

import torch


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    return wrapper
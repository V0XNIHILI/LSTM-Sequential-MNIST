from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def evaluating(net: nn.Module):
    """Base code taken from: https://discuss.pytorch.org/t/opinion-eval-should-
    be-a-context-manager/18998/2.

    Args:
        net (nn.Module): Net that will be evaluated

    Yields:
        nn.Module: Net to be evaluated
    """

    is_train = net.training

    try:
        net.eval()

        yield net
    finally:
        if is_train:
            net.train()


@contextmanager
def training(net: nn.Module):
    """Base code taken from: https://discuss.pytorch.org/t/opinion-eval-should-
    be-a-context-manager/18998/2.

    Args:
        net (nn.Module): Net that will be trained

    Yields:
        nn.Module: Net to be trained
    """

    is_eval = not net.training

    try:
        net.train()

        yield net
    finally:
        if is_eval:
            net.eval()

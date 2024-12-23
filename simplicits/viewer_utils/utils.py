"""
Modified from Shape of Motion (https://github.com/vye16/shape-of-motion).
"""

import colorsys
from typing import cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from viser import ViserServer


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class VisManager(metaclass=Singleton):
    _servers = {}


def get_server(port: int | None = None) -> ViserServer:
    manager = VisManager()
    if port is None:
        avail_ports = list(manager._servers.keys())
        port = avail_ports[0] if len(avail_ports) > 0 else 8890
    if port not in manager._servers:
        manager._servers[port] = ViserServer(port=port, verbose=False)
    return manager._servers[port]

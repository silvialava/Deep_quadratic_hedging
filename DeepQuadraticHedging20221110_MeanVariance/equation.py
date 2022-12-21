#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:22:18 2022

@author: Alessandro Gnoatto
"""

import numpy as np


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config['dim']# m > 0 (in Lim's notation)
        self.dim_nohedge = eqn_config['dim_nohedge'] # d >= 0 (in Lim's notation)
        self.total_time = eqn_config['total_time']
        self.num_time_interval = eqn_config['num_time_interval']
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z, v, zv):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x, v):
        """Terminal condition of the PDE."""
        raise NotImplementedError
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:06:16 2021

@author: yui
"""

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

predPath = './result/000010.tif'
atomPath = './atom/000010.tif'

OD = io.imread(predPath) - io.imread(atomPath)
OD = np.asarray(OD, dtype=np.uint32)

io.imsave('svd_pred.tif', OD)
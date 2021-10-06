# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:44:17 2021

@author: yui
"""

#from test.init_test import *

from pycoldatom.functions.fringe import FringeRemove
from pycoldatom.functions.refimages import add_noise

import os, sys

import numpy as np
from skimage import io
import os

from numpy import linalg as lg
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


a = 3/16
b = -13/16

basisPath = './basis/'
basisOri = []

TrainFlag = 0

for i in os.listdir(basisPath):
    basisName = os.path.join(basisPath, i)
    basisFig = io.imread(basisName)
    basisFig = ((basisFig / 4294967295) - b) / a
    basisFig = np.exp(basisFig)
    basisFig = np.reshape(basisFig, (-1))
    basisOri.append(basisFig)
print('Load basis dataset')


trainNum = len(basisOri)

trainSet = []
basisSet = []
trainset_avg_field = np.zeros_like(basisOri[0])
for i in basisOri:
    trainset_avg = np.mean(i)
    figI = i - trainset_avg - trainset_avg_field
    trainSet.append(figI)
    trainset_avg_field += figI

trainset_avg_field /= len(basisOri)
trainSet = np.asarray(trainSet) - trainset_avg_field # (20, 226576)
trainSet = trainSet.T
#print(trainSet.shape)

remover = FringeRemove(trunc=1e-10)

for i in range(trainNum):
    trainI = trainSet[:, i]
    remover.updateLibrary(trainI)
    U, s, V = remover.svd_result
    
mask = np.zeros((476, 476))
for i in range(476):
    for j in range(476):
        if (i-238)**2+(j-238)**2 < 95**2:
            mask[i,j] = 0 # 1 mask, 0 still
maskHot = np.reshape(mask, (-1))  

for i in os.listdir('./atom/'):    
        
    sample_data = os.path.join('./atom/', i)
    atomFig = io.imread(sample_data)
    atomFig = ((atomFig / 4294967295) - b) / a
    atomFig = np.exp(atomFig)
    atomFig = np.reshape(atomFig, (-1))
    curr_img_mean = np.mean(atomFig)
    atomFig = atomFig - curr_img_mean
    
    atomMask = atomFig * (1-maskHot)
    
    x, data2 = remover.reconstruct(atomMask)
    scale_fac = lg.norm(atomMask)/lg.norm(data2)    
    
    curr_img = data2*scale_fac
    curr_img = np.squeeze(curr_img)
    curr_img = curr_img + curr_img_mean + trainset_avg_field
    
    curr_img = np.reshape(curr_img, (476, 476))
    
    resultDir = './result/'
    curr_img = np.log(curr_img)*a+b
    curr_img = curr_img * (2**32-1)
    curr_img = np.asarray(curr_img, dtype=np.uint32)
    #plt.imshow(curr_img, cmap='gray')
    io.imsave(os.path.join(resultDir, i), curr_img)
    

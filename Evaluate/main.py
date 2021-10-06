import numpy as np
import re
import sys
import os

#import vifp
import ssim
#import ssim_theano
import psnr
#import niqe
#import reco
from skimage import io

model_name = 'OptimizedPCA'
refDir = './data/R_with_atoms/'
predDir = './result/{0}/'.format(model_name)

ssimList = []
psnrList = []

count_num = 0
count_len = len(os.listdir(predDir))

for i in os.listdir(predDir):
    predPath = os.path.join(predDir, i)
    refPath = os.path.join(refDir, i)
    predImg = io.imread(predPath)
    predImg = np.asarray(predImg)
    predImg = np.squeeze(predImg)
    if 'WGAN' in model_name: # 384*384
        refImg = io.imread(refPath)
        #refImg = refImg[46:430, 46:430]
        #print('Pred img', predImg.shape)
        predImg = predImg[84:276, 84:276]
        refImg = refImg[142:334, 142:334]
        #refImg = refImg[46:430, 46:430]
        #print('Pred img', predImg.shape)
    elif 'UNet' in model_name:
        refImg = io.imread(refPath)
        refImg = refImg[143:333, 143:333]
    else:
        predImg = predImg[143:333, 143:333]
        refImg = io.imread(refPath)[143:333, 143:333]
        #print('Error :')
    
    refImg = np.array(refImg)
    
    predImg = predImg / 4294967295 * 255
    refImg = refImg / 4294967295 * 255
    #predImg = np.asarray(predImg, np.uint8)
    #refImg = np.asarray(refImg, np.uint8)
    
    ssimScore = ssim.ssim(predImg, refImg)
    psnrScore = psnr.psnr(predImg, refImg)
    
    ssimList.append(ssimScore)
    psnrList.append(psnrScore)
    #print('PSNR ', ssimScore)
    count_num += 1
    #print('[{1}/{2}]PSNR {0:.4f}'.format(psnrScore, count_num, count_len))

print('Method {0} SSIM {1:.4f} ; PSNR {2:.4f}'.format(model_name,\
                                              np.mean(ssimList),\
                                                  np.mean(psnrList)))


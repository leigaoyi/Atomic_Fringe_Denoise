import numpy as np
from skimage import io
import os

from numpy import linalg as lg
import matplotlib.pyplot as plt

a = 3/16
b = -13/16

basisPath = './basis/'
basisOri = []

TrainFlag = 1

for i in os.listdir(basisPath)[:150]:
    basisName = os.path.join(basisPath, i)
    basisFig = io.imread(basisName)
    basisFig = ((basisFig / 4294967295) - b) / a
    basisFig = np.exp(basisFig)
    basisFig = np.reshape(basisFig, (-1))
    basisOri.append(basisFig)
print('Load basis dataset')

if TrainFlag:
     
      print(basisOri[1].shape)
      
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
      print(trainSet.shape)
      
      U, sigma, VT = lg.svd(trainSet.T,full_matrices=0)
      print('[SVD] over!')
      eigvals_vec = np.diag(sigma)
      basisMats = np.zeros((476 * 476, len(basisOri)))
      C = np.zeros((trainNum,trainNum))
      pinv = lg.pinv(trainSet.T)
      for i in range(sigma.shape[0]):
            v_i = U[:, i]
            C[:, i] = np.matmul(pinv , v_i[:, np.newaxis])[:, 0]
            ## C shape (20, 20)
            
            # for j in range(sigma.shape[0]):
            #       basisMats[:, i] = basisMats[:, i] + trainSet.T[:, j]* C[j, i]
            # print('[{0}/{1}] Basis'.format(i, trainNum))
      basisMats = np.matmul(trainSet.T, C)
            
      print('[Basis calculate]')                  
      #np.save('PCA.npy', [U, basisMats, trainset_avg_field], allow_pickle=True)
      with open('PCA.npy', 'wb') as f:
          np.save(f, U)
          np.save(f, basisMats)
          np.save(f, trainset_avg_field)
      
else:
      #U, basisMats, trainset_avg_field = np.load('PCA.npy', allow_pickle=True)
      with open('PCA.npy', 'rb') as f:
          U = np.load(f)
          basisMats = np.load(f)
          trainset_avg_field = np.load(f)
      print('Load npy successful')    
      trainNum = U.shape[1]

# sampleFig = basisOri[12]

# curr_img_rec = np.zeros(476*476)
      
# for i in range(trainNum):
#       wij = sampleFig * U[:, i]
#       curr_img_rec = curr_img_rec + wij*basisMats[:, i]
      
# scale_fac = np.mean(sampleFig)/np.mean(curr_img_rec)      
# curr_img = np.reshape(curr_img_rec*scale_fac, (476, 476))
# curr_img = np.log(curr_img)*a+b
# curr_img = curr_img * (2**32-1)
# curr_img = np.asarray(curr_img, dtype=np.uint32)
# plt.imshow(curr_img, cmap='gray')
# io.imsave('./tmp/test.tif', curr_img)

## process dir

atomDir = './atom_test/'
resultDir = './result/'

if not os.path.exists(resultDir):
    os.makedirs(resultDir)
count = 0
countNum = len(os.listdir(atomDir))

mask = np.zeros((476, 476))
for i in range(476):
    for j in range(476):
        if (i-238)**2+(j-238)**2 < 95**2:
            mask[i,j] = 1
maskHot = np.reshape(mask, (-1))        

for atomR in os.listdir(atomDir):
    atomName = os.path.join(atomDir, atomR)
    atomFig = io.imread(atomName)
    atomFig = ((atomFig / 4294967295) - b) / a
    atomFig = np.exp(atomFig)
    atomFig = np.reshape(atomFig, (-1))
    curr_img_mean = np.mean(atomFig)
    atomFig = atomFig - curr_img_mean
    
    curr_img_rec = np.zeros(476*476)
          
    for i in range(trainNum):
          wij = np.sum((atomFig*(1-maskHot)) * U[:, i])
          curr_img_rec = curr_img_rec + wij*basisMats[:, i]
          
    atomMask = atomFig * (1-maskHot)
    currMask = curr_img_rec * (1-maskHot)
          
    scale_fac = lg.norm(atomMask)/lg.norm(currMask)    
    
    curr_img = curr_img_rec*scale_fac
    curr_img = curr_img + curr_img_mean + trainset_avg_field
    curr_img = np.reshape(curr_img, (476, 476))
    
    curr_img = np.log(curr_img)*a+b
    curr_img = curr_img * (2**32-1)
    curr_img = np.asarray(curr_img, dtype=np.uint32)
    #plt.imshow(curr_img, cmap='gray')
    io.imsave(os.path.join(resultDir, atomR), curr_img)
    print('[{0}/{1}] --Reconstruct'.format(count, countNum)) 
    count += 1

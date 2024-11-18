import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy
import numpy as np
#import cv2
from utils import meanPerSlice
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import os
import math

import torch
import torch.optim as optim

from torch import nn
from utils import trainModel
from utils import reshapeDataSet
from unet import Unet
from utils import saveDataCsv

from unet import UnetWithResidual5Layers
from unet import UnetWithResidual
from unet import Unet512
from unet import UnetDe1a16Hasta512

def lossFunction(output,target):
    loss = (torch.sum((output - target)**2)) / torch.sum(target)
    return loss

######################### CHECK DEVICE ######################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

######################## TRAINING PARAMETERS ###############
batchSize = 8
epochs = 100
learning_rate=0.00001
#learning_rate=0.0001
printStep_epochs = 1
plotStep_epochs = 5
printStep_batches = 100
plotStep_batches = math.inf

# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../data/BrainWebSimulations2D/'
lowDose_perc = 5
actScaleFactor = 100/lowDose_perc

#pathGroundTruth = path+'/100'
#arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

pathNoisyDataSet = path + str(lowDose_perc)
#arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []
nametrainNoisyDataSet = []

## Type of normalization ##
normMeanStd = True
normMean = False
normMax = False
normBrainMean = False
normBrainMeanWithErode = False
normGreyMatterMean = False
randomScaleGaussian = False
randomScale = False

nameThisNet = 'UnetDe1a32Hasta512_MSE_lr1e-05_AlignTrue_MeanSTD'.format(learning_rate)

outputPath = '../../results/newContrast/' + nameThisNet + '/'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../data/BrainWebSimulations2D'


# Choosing UNet for training
#unet = Unet(1,1)
#unet = UnetWithResidual(1, 1)
#unet = UnetDe1a16Hasta512(1,1,16)
unet = Unet512(1,1,32)

#unet = UnetWithResidual(1,1)
#unet = UnetWithResidual5Layers(1,1)



trainGroundTruth = sitk.ReadImage(path+'/trainGroundTruth.nii')
voxelSize_mm = trainGroundTruth.GetSpacing()
trainGroundTruth = sitk.GetArrayFromImage(trainGroundTruth)

trainNoisyDataSet = sitk.ReadImage(path+'/trainNoisyDataSet.nii')
trainNoisyDataSet = sitk.GetArrayFromImage(trainNoisyDataSet)

trainGreyMask = sitk.ReadImage(path+'/trainGreyMatterMask.nii')
trainGreyMask = sitk.GetArrayFromImage(trainGreyMask)

trainWhiteMask = sitk.ReadImage(path+'/trainWhiteMatterMask.nii')
trainWhiteMask = sitk.GetArrayFromImage(trainWhiteMask)

validGroundTruth = sitk.ReadImage(path+'/validGroundTruth.nii')
validGroundTruth = sitk.GetArrayFromImage(validGroundTruth)

validNoisyDataSet = sitk.ReadImage(path+'/validNoisyDataset.nii')
validNoisyDataSet = sitk.GetArrayFromImage(validNoisyDataSet)

validGreyMask = sitk.ReadImage(path+'/validGreyMatterMask.nii')
validGreyMask = sitk.GetArrayFromImage(validGreyMask)

validWhiteMask = sitk.ReadImage(path+'/validWhiteMatterMask.nii')
validWhiteMask = sitk.GetArrayFromImage(validWhiteMask)

trainGroundTruth = reshapeDataSet(trainGroundTruth)
trainNoisyDataSet = reshapeDataSet(trainNoisyDataSet)
trainGreyMask = reshapeDataSet(trainGreyMask)
trainWhiteMask = reshapeDataSet(trainWhiteMask)
validGroundTruth = reshapeDataSet(validGroundTruth)
validNoisyDataSet = reshapeDataSet(validNoisyDataSet)
validGreyMask = reshapeDataSet(validGreyMask)
validWhiteMask = reshapeDataSet(validWhiteMask)

if normMean:
    meanSliceTrainNoisy = np.mean(trainNoisyDataSet, axis=(1, 2, 3))
    meanSliceTrainGround = np.mean(trainGroundTruth, axis=(1, 2, 3))

    meanSliceValidNoisy = np.mean(validNoisyDataSet, axis=(1, 2, 3))
    meanSliceValidGround = np.mean(validGroundTruth, axis=(1, 2, 3))

    sliceNormTrainNoisy = trainNoisyDataSet [:,:,:, :]/ meanSliceTrainNoisy[:, None,None, None]
    sliceNormTrainGT = trainGroundTruth [:,:,:, :] /meanSliceTrainGround[:, None,None, None]
    sliceNormValidNoisy = validNoisyDataSet [:,:,:, :]/ meanSliceValidNoisy[:, None,None, None]
    sliceNormValidGT = validGroundTruth [:,:,:, :]/ meanSliceValidGround[:, None,None, None]

# if normBrainMeanWithErode:
#     BrainMask = validWhiteMask + validGreyMask
#     BrainMask = np.where(BrainMask != 0, 1, 0)
#
#     kernel = np.ones((3, 3), np.uint8)
#     imagen_erosionada = cv2.erode((BrainMask[:,0,:,:]).astype(np.uint8), kernel, iterations=1)
#
#     #meanValidNoisy= meanPerSlice((validNoisyDataSet[:,0,:,:]* BrainMask[:,0,:,:]))
#     #meanValidGroundTruth = meanPerSlice((validGroundTruth[:,0,:,:] * BrainMask[:,0,:,:]))
#
#     meanValidNoisy = meanPerSlice((validNoisyDataSet[:, 0, :, :] * imagen_erosionada.astype(np.float32)))
#     meanValidGroundTruth = meanPerSlice((validGroundTruth[:, 0, :, :] * imagen_erosionada.astype(np.float32)))
#
#     sliceNormValidNoisy = []
#     sliceNormValidGT = []
#
#     for meanIdx in range(0,len(meanValidNoisy)):
#         if meanValidNoisy[meanIdx]!=0 and meanValidGroundTruth[meanIdx]!=0:
#             sliceNormValidNoisy.append((validNoisyDataSet[meanIdx, :, :, :] / meanValidNoisy[meanIdx, None, None, None]).astype(np.float32))
#             sliceNormValidGT.append((validGroundTruth[meanIdx, :, :, :] / meanValidGroundTruth[meanIdx, None, None, None]).astype(np.float32))
#
#     BrainMask = trainWhiteMask + trainGreyMask
#     BrainMask = np.where(BrainMask != 0, 1, 0)
#     imagen_erosionada = cv2.erode((BrainMask[:, 0, :, :]).astype(np.uint8), kernel, iterations=1)
#
#     #meanTrainNoisy = meanPerSlice((trainNoisyDataSet[:,0,:,:] * BrainMask[:,0,:,:]))
#     #meanTrainGroundTruth = meanPerSlice((trainGroundTruth[:,0,:,:] * BrainMask[:,0,:,:]))
#
#     meanTrainNoisy = meanPerSlice((trainNoisyDataSet[:, 0, :, :] * imagen_erosionada.astype(np.float32)))
#     meanTrainGroundTruth = meanPerSlice((trainGroundTruth[:, 0, :, :] * imagen_erosionada.astype(np.float32)))
#
#     sliceNormTrainNoisy = []
#     sliceNormTrainGT = []
#
#     for meanIdx in range(0,len(meanTrainNoisy)):
#         if meanTrainNoisy[meanIdx]!=0 and meanTrainGroundTruth[meanIdx]!=0:
#             sliceNormTrainNoisy.append((trainNoisyDataSet[meanIdx, :, :, :] / meanTrainNoisy[meanIdx, None, None, None]).astype(np.float32))
#             sliceNormTrainGT.append((trainGroundTruth[meanIdx, :, :, :] / meanTrainGroundTruth[meanIdx, None, None, None]).astype(np.float32))

if normBrainMean:
    sliceNormValidNoisy = []
    sliceNormValidGT = []
    sliceNormTrainNoisy = []
    sliceNormTrainGT = []

    BrainMask = validWhiteMask + validGreyMask
    BrainMask = np.where(BrainMask != 0, 1, 0)

    meanValidNoisy= meanPerSlice((validNoisyDataSet[:,0,:,:]* BrainMask[:,0,:,:]))
    meanValidGroundTruth = meanPerSlice((validGroundTruth[:,0,:,:] * BrainMask[:,0,:,:]))

    for idx in range(0,len(meanValidNoisy)):
        if meanValidNoisy[idx] > 0.4 and meanValidGroundTruth[idx] > 0.4:
            sliceNormValidNoisy.append((validNoisyDataSet[idx, :, :, :] / meanValidNoisy[idx, None, None, None]).astype(np.float32))
            sliceNormValidGT.append((validGroundTruth[idx, :, :, :] / meanValidGroundTruth[idx, None, None, None]).astype(np.float32))


    BrainMask = trainWhiteMask + trainGreyMask
    BrainMask = np.where(BrainMask != 0, 1, 0)

    meanTrainNoisy = meanPerSlice((trainNoisyDataSet[:,0,:,:] * BrainMask[:,0,:,:]))
    meanTrainGroundTruth = meanPerSlice((trainGroundTruth[:,0,:,:] * BrainMask[:,0,:,:]))

    for idx in range(0,len(meanTrainNoisy)):
        if meanTrainNoisy[idx] > 0.4 and meanTrainGroundTruth[idx] > 0.4:
            sliceNormTrainNoisy.append((trainNoisyDataSet[idx, :, :, :] / meanTrainNoisy[idx, None, None, None]).astype(np.float32))
            sliceNormTrainGT.append((trainGroundTruth[idx, :, :, :] / meanTrainGroundTruth[idx, None, None, None]).astype(np.float32))


if normGreyMatterMean:
    BrainMask = np.where(validGreyMask != 0, 1, 0)

    meanValidNoisy= meanPerSlice(validNoisyDataSet[:,0,:,:]* BrainMask[:,0,:,:])
    meanValidGroundTruth = meanPerSlice(validGroundTruth[:,0,:,:] * BrainMask[:,0,:,:])

    BrainMask = np.where(trainGreyMask != 0, 1, 0)

    meanTrainNoisy = meanPerSlice(trainNoisyDataSet[:,0,:,:] * BrainMask[:,0,:,:])
    meanTrainGroundTruth = meanPerSlice(trainGroundTruth[:,0,:,:] * BrainMask[:,0,:,:])

    sliceNormTrainNoisy = (trainNoisyDataSet[:, :, :, :] / meanTrainNoisy[:, None, None, None]).astype(np.float32)
    sliceNormTrainGT = (trainGroundTruth[:, :, :, :] / meanTrainGroundTruth[:, None, None, None]).astype(np.float32)
    sliceNormValidNoisy = (validNoisyDataSet[:, :, :, :] / meanValidNoisy[:, None, None, None]).astype(np.float32)
    sliceNormValidGT = (validGroundTruth[:, :, :, :] / meanValidGroundTruth[:, None, None, None]).astype(np.float32)

    sliceNormTrainNoisy = np.nan_to_num(sliceNormTrainNoisy)
    sliceNormTrainGT = np.nan_to_num(sliceNormTrainGT)
    sliceNormValidNoisy = np.nan_to_num(sliceNormValidNoisy)
    sliceNormValidGT = np.nan_to_num(sliceNormValidGT)

if normMax:
    maxSliceTrainNoisy = np.max(trainNoisyDataSet, axis=(1, 2, 3))
    maxSliceTrainGround = np.max(trainGroundTruth, axis=(1, 2, 3))

    maxSliceValidNoisy = np.max(validNoisyDataSet,axis=(1, 2, 3))
    maxSliceValidGround = np.max(validGroundTruth, axis=(1, 2, 3))

    sliceNormTrainNoisy = trainNoisyDataSet [:,:,:, :]/ maxSliceTrainNoisy[:, None,None, None]
    sliceNormTrainGT = trainGroundTruth [:,:,:, :] /maxSliceTrainGround[:, None,None, None]
    sliceNormValidNoisy = validNoisyDataSet [:,:,:, :]/ maxSliceValidNoisy[:, None,None, None]
    sliceNormValidGT = validGroundTruth [:,:,:, :]/ maxSliceValidGround[:, None,None, None]

if normMeanStd:
    meanSliceTrainNoisy = np.mean(trainNoisyDataSet, axis=(1, 2, 3))
    meanSliceTrainGround = np.mean(trainGroundTruth, axis=(1, 2, 3))

    meanSliceValidNoisy = np.mean(validNoisyDataSet, axis=(1, 2, 3))
    meanSliceValidGround = np.mean(validGroundTruth, axis=(1, 2, 3))

    stdSliceTrainNoisy = np.std(trainNoisyDataSet, axis=(1, 2, 3))
    stdSliceTrainGround = np.std(trainGroundTruth, axis=(1, 2, 3))

    stdSliceValidNoisy = np.std(validNoisyDataSet, axis=(1, 2, 3))
    stdSliceValidGround = np.std(validGroundTruth, axis=(1, 2, 3))

    sliceNormTrainNoisy = (trainNoisyDataSet [:,:,:, :]- meanSliceTrainNoisy[:, None,None, None]) / stdSliceTrainNoisy[:, None,None, None]
    sliceNormTrainGT = (trainGroundTruth [:,:,:, :]- meanSliceTrainGround[:, None,None, None]) / stdSliceTrainGround[:, None,None, None]

    sliceNormValidNoisy = (validNoisyDataSet [:,:,:, :]- meanSliceValidNoisy[:, None,None, None]) / stdSliceValidNoisy[:, None,None, None]
    sliceNormValidGT = (validGroundTruth [:,:,:, :]- meanSliceValidGround[:, None,None, None]) / stdSliceValidGround[:, None,None, None]

if normMeanStd == False and normMean == False and normMax == False and normGreyMatterMean == False and normBrainMean == False:
    sliceNormTrainNoisy = trainNoisyDataSet
    sliceNormTrainGT = trainGroundTruth
    sliceNormValidNoisy = validNoisyDataSet
    sliceNormValidGT = validGroundTruth

sliceNormTrainNoisy = np.array(sliceNormTrainNoisy)
sliceNormTrainGT = np.array(sliceNormTrainGT)
sliceNormValidNoisy = np.array(sliceNormValidNoisy)
sliceNormValidGT = np.array(sliceNormValidGT)


# Shuffle the data:
sliceNormTrainNoisy, sliceNormTrainGT = shuffle(sliceNormTrainNoisy, sliceNormTrainGT, random_state=0)
sliceNormValidNoisy, sliceNormValidGT = shuffle(sliceNormValidNoisy, sliceNormValidGT, random_state=0)

print('Valid GT:',sliceNormValidGT.shape)
print('Train GT:',sliceNormTrainGT.shape)

print('Train Noisy Dataset:',sliceNormTrainNoisy.shape)
print('Valid Noisy Dataset:',sliceNormValidNoisy.shape)

# Create dictionaries with training sets:
trainingSet = dict([('input',sliceNormTrainNoisy), ('output', sliceNormTrainGT)])
validSet = dict([('input',sliceNormValidNoisy), ('output', sliceNormValidGT)])

print('Data set size. Training set: {0}. Valid set: {1}.'.format(trainingSet['input'].shape[0], validSet['input'].shape[0]))

# Entrenamiento #
# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)




lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer, batchSize,
                                                                                            epochs, device, pre_trained = False, save = True, outputPath=outputPath,name = nameThisNet,
                                                                                            printStep_batches = printStep_batches, plotStep_batches = plotStep_batches,
                                                                                            printStep_epochs = printStep_epochs, plotStep_epochs = plotStep_epochs)
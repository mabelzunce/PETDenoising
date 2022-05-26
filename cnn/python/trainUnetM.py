import nibabel as nb
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from datetime import datetime
from utils import trainModel
from utils import reshapeDataSet
from unetM import Unet

unet = Unet()

# Importo base de datos ...
path = os.getcwd()
pathGroundTruth = path+'/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

ramdomIdx = np.random.randint(1, len(arrayGroundTruth)+1, 2).tolist()

for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    noisyDataSet = sitk.ReadImage(pathGroundTruthElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    name = element[23:-4]
    if name not in str(ramdomIdx):
        trainGroundTruth.append(noisyDataSet)
        # globals()['GT%s' % element] = reshapeDataSet(noisyDataSet)
    else:
        validGroundTruth.append(noisyDataSet)

pathNoisyDataSet = path+'/noisyDataSet/1'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []

for element in arrayNoisyDataSet:
    pathNoisyDataSetElement = pathNoisyDataSet+'/'+element
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    name = element[21:-4]
    if name not in str(ramdomIdx):
        trainNoisyDataSet.append(noisyDataSet)
        # globals()['GT%s' % element] = reshapeDataSet(noisyDataSet)
    else:
        validNoisyDataSet.append(noisyDataSet)

## Normalizo los arrays
trainGroundTruthNorm = []
for subject in range(0,len(trainGroundTruth)):
    subjectElement = trainGroundTruth[subject]
    for slice in range(0,subjectElement.shape[0]):
        meanSlice = subjectElement[slice,:,:].mean()
        if meanSlice == 0.0:
            norm = ((subjectElement[slice, :, :] ))
        else:
            norm = ((subjectElement[slice,:,:] / meanSlice))
        trainGroundTruthNorm.append(norm)

validGroundTruthNorm = []
for subject in range(0, len(validGroundTruth)):
    subjectElement = validGroundTruth[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        if meanSlice == 0.0:
            norm = ((subjectElement[slice, :, :]))
        else:
            norm = ((subjectElement[slice, :, :] / meanSlice))
        trainGroundTruthNorm.append(norm)
        validGroundTruthNorm.append(norm)

trainNoisyDataSetNorm = []
for subject in range(0, len(trainNoisyDataSet)):
    subjectElement = trainNoisyDataSet[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        if meanSlice == 0.0:
            norm = ((subjectElement[slice, :, :]))
        else:
            norm = ((subjectElement[slice, :, :] / meanSlice))
        trainGroundTruthNorm.append(norm)
        trainNoisyDataSetNorm.append(norm)

validNoisyDataSetNorm = []
for subject in range(0, len(validNoisyDataSet)):
    subjectElement = validNoisyDataSet[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        if meanSlice == 0.0:
            norm = ((subjectElement[slice, :, :]))
        else:
            norm = ((subjectElement[slice, :, :] / meanSlice))
        trainGroundTruthNorm.append(norm)e)
        validNoisyDataSetNorm.append(norm)

trainGroundTruthNorm = np.array(trainGroundTruthNorm)
validGroundTruthNorm = np.array(validGroundTruthNorm)
trainNoisyDataSetNorm = np.array(trainNoisyDataSetNorm)
validNoisyDataSetNorm = np.array(validNoisyDataSetNorm)

trainGroundTruthNorm = reshapeDataSet(trainGroundTruthNorm)
validGroundTruthNorm = reshapeDataSet(validGroundTruthNorm)
trainNoisyDataSetNorm = reshapeDataSet(trainNoisyDataSetNorm)
validNoisyDataSetNorm = reshapeDataSet(validNoisyDataSetNorm)

# Create dictionaries with training sets:
trainingSet = dict([('input',trainNoisyDataSetNorm), ('output', trainGroundTruthNorm)])
validSet = dict([('input',validNoisyDataSetNorm), ('output', validGroundTruthNorm)])

print('Data set size. Training set: {0}. Valid set: {1}.'.format(trainingSet['input'].shape[0], validSet['input'].shape[0]))

# Entrenamiento #
# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer,4,70)

df = pd.DataFrame(lossValuesTraining)
df.to_excel('lossValuesTrainingSetBatchModel3.xlsx')

df = pd.DataFrame(lossValuesEpoch)
df.to_excel('lossValuesTrainingSetEpochModel3.xlsx')

df = pd.DataFrame(lossValuesDevSet)
df.to_excel('lossValuesDevSetBatchModel3.xlsx')

df = pd.DataFrame(lossValuesDevSetAllEpoch)
df.to_excel('lossValuesDevSetEpochModel3.xlsx')
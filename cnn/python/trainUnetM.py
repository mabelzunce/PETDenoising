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

from datetime import datetime
from utils import trainModel
from utils import reshapeDataSet
from unetM import Unet


# Importo base de datos ...
path = os.getcwd()
pathGroundTruth = path+'/NewDataset/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

unet = Unet()

ramdomIdx = np.random.randint(1, len(arrayGroundTruth)+1, 2).tolist()
print(ramdomIdx)

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

pathNoisyDataSet = path+'/NewDataset/noisyDataSet/5'
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
        trainGroundTruthNorm.append(np.rot90(norm))

validGroundTruthNorm = []
for subject in range(0, len(validGroundTruth)):
    subjectElement = validGroundTruth[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        stdSlice = subjectElement[slice, :, :].std()
        if stdSlice == 0.0:
            norm = ((subjectElement[slice, :, :]) - meanSlice)
        else:
            norm = ((subjectElement[slice, :, :] - meanSlice) / stdSlice)
        validGroundTruthNorm.append(norm)
        validGroundTruthNorm.append(np.rot90(norm))

trainNoisyDataSetNorm = []
for subject in range(0, len(trainNoisyDataSet)):
    subjectElement = trainNoisyDataSet[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        stdSlice = subjectElement[slice, :, :].std()
        if stdSlice == 0.0:
            norm = ((subjectElement[slice, :, :]) - meanSlice)
        else:
            norm = ((subjectElement[slice, :, :] - meanSlice) / stdSlice)
        trainNoisyDataSetNorm.append(norm)
        trainNoisyDataSetNorm.append(np.rot90(norm))

validNoisyDataSetNorm = []
for subject in range(0, len(validNoisyDataSet)):
    subjectElement = validNoisyDataSet[subject]
    for slice in range(0, subjectElement.shape[0]):
        meanSlice = subjectElement[slice, :, :].mean()
        stdSlice = subjectElement[slice, :, :].std()
        if stdSlice == 0.0:
            norm = ((subjectElement[slice, :, :]) - meanSlice)
        else:
            norm = ((subjectElement[slice, :, :] - meanSlice) / stdSlice)
        validNoisyDataSetNorm.append(norm)
        validNoisyDataSetNorm.append(np.rot90(norm))

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
df.to_excel('lossValuesTrainingSetBatchModel5.xlsx')

df = pd.DataFrame(lossValuesEpoch)
df.to_excel('lossValuesTrainingSetEpochModel5.xlsx')

df = pd.DataFrame(lossValuesDevSet)
df.to_excel('lossValuesDevSetBatchModel5.xlsx')

df = pd.DataFrame(lossValuesDevSetAllEpoch)
df.to_excel('lossValuesDevSetEpochModel5.xlsx')

df = pd.DataFrame(ramdomIdx)
df.to_excel('validSubjects.xlsx')
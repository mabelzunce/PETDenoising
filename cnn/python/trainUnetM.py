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

nameGroundTruth = []
for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    noisyDataSet = sitk.ReadImage(pathGroundTruthElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    name = element[23:-4]
    nameGroundTruth.append(name)
    if name not in str(ramdomIdx):
        trainGroundTruth.append(noisyDataSet)
    else:
        validGroundTruth.append(noisyDataSet)

pathNoisyDataSet = path+'/NewDataset/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []
nametrainNoisyDataSet = []
for element in arrayNoisyDataSet:
    pathNoisyDataSetElement = pathNoisyDataSet+'/'+element
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    name = element[21:-4]
    nametrainNoisyDataSet.append(name)
    if name not in str(ramdomIdx):
        trainNoisyDataSet.append(noisyDataSet)
    else:
        validNoisyDataSet.append(noisyDataSet)


## Set de entramiento
trainNoisyDataSetNorm = []
trainGroundTruthNorm = []
for subject in range(0, len(trainNoisyDataSet)):
    subjectElementNoisy = trainNoisyDataSet[subject]
    subjectElementGroundTruth = trainGroundTruth[subject]
    for slice in range(0, subjectElementNoisy.shape[0]):
        maxSliceNoisy = subjectElementNoisy[slice, :, :].mean()
        maxSliceGroundTruth = subjectElementGroundTruth[slice, :, :].mean()
        if (maxSliceNoisy > 0.0000001) and (maxSliceGroundTruth > 0.0) :
            normNoisy = ((subjectElementNoisy[slice, :, :]) / maxSliceNoisy)
            trainNoisyDataSetNorm.append(normNoisy)
            trainNoisyDataSetNorm.append(np.rot90(normNoisy))
            normGroundTruth = ((subjectElementGroundTruth[slice, :, :]) / maxSliceGroundTruth)
            trainGroundTruthNorm.append(normGroundTruth )
            trainGroundTruthNorm.append(np.rot90(normGroundTruth))

#for subject in range(0,len(trainGroundTruth)):
    #subjectElement = trainGroundTruth[subject]
    #for slice in range(0,subjectElement.shape[0]):
        #maxSlice = subjectElement[slice,:,:].max()
        #if (maxSlice > 0.0):
            #norm = ((subjectElement[slice,:,:] / maxSlice))
            #trainGroundTruthNorm.append(norm)
            #trainGroundTruthNorm.append(np.rot90(norm))

# Set de validacion
validNoisyDataSetNorm = []
validGroundTruthNorm = []
for subject in range(0, len(validNoisyDataSet)):
    subjectElementNoisy = validNoisyDataSet[subject]
    subjectElementGroundTruth = validGroundTruth[subject]
    for slice in range(0, subjectElementNoisy.shape[0]):
        maxSliceNoisy = subjectElementNoisy[slice, :, :].max()
        maxSliceGroundTruth = subjectElementGroundTruth[slice, :, :].max()
        if (maxSliceNoisy > 0.0000001) and (maxSliceGroundTruth > 0.0) :
            normNoisy = ((subjectElementNoisy[slice, :, :]) / maxSliceNoisy)
            validNoisyDataSetNorm.append(normNoisy)
            validNoisyDataSetNorm.append(np.rot90(normNoisy))
            normGroundTruth = ((subjectElementGroundTruth[slice, :, :]) / maxSliceGroundTruth)
            validGroundTruthNorm.append(normGroundTruth)
            validGroundTruthNorm.append(np.rot90(normGroundTruth))

#for subject in range(0, len(validGroundTruth)):
#    subjectElement = validGroundTruth[subject]
#    for slice in range(0, subjectElement.shape[0]):
#        maxSlice = subjectElement[slice, :, :].max()
#        if (maxSlice > 0.0):
#            norm = ((subjectElement[slice, :, :] / maxSlice))
#            validGroundTruthNorm.append(norm)
            #validGroundTruthNorm.append(np.rot90(norm))

trainGroundTruthNorm = np.array(trainGroundTruthNorm)
validGroundTruthNorm = np.array(validGroundTruthNorm)
trainNoisyDataSetNorm = np.array(trainNoisyDataSetNorm)
validNoisyDataSetNorm = np.array(validNoisyDataSetNorm)

trainGroundTruthNorm = reshapeDataSet(trainGroundTruthNorm)
validGroundTruthNorm = reshapeDataSet(validGroundTruthNorm)
trainNoisyDataSetNorm = reshapeDataSet(trainNoisyDataSetNorm)
validNoisyDataSetNorm = reshapeDataSet(validNoisyDataSetNorm)

print('Train GT:',trainGroundTruthNorm.shape)
print('Valid GT:',validGroundTruthNorm.shape)

print('Train Noisy Dataset:',trainNoisyDataSetNorm.shape)
print('Valid Noisy Dataset:',validNoisyDataSetNorm.shape)

df = pd.DataFrame(ramdomIdx)
df.to_excel('validSubjectsModel6.xlsx')

# Create dictionaries with training sets:
trainingSet = dict([('input',trainNoisyDataSetNorm), ('output', trainGroundTruthNorm)])
validSet = dict([('input',validNoisyDataSetNorm), ('output', validGroundTruthNorm)])

print('Data set size. Training set: {0}. Valid set: {1}.'.format(trainingSet['input'].shape[0], validSet['input'].shape[0]))

# Entrenamiento #
# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer,4,70,save = True, name = 'Model6')

df = pd.DataFrame(lossValuesTraining)
df.to_excel('lossValuesTrainingSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesEpoch)
df.to_excel('lossValuesTrainingSetEpochModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSet)
df.to_excel('lossValuesDevSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSetAllEpoch)
df.to_excel('lossValuesDevSetEpochModel6Total.xlsx')

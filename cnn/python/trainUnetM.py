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

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

######################## TRAINING PARAMETERS ###############
batchSize = 4
epochs = 70


# Importo base de datos ...
path = os.getcwd() #'D:/UNSAM/PET/BrainWebSimulations/'
#pathGroundTruth = path+'/100'
pathGroundTruth = path+'/NewDataset/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

#pathNoisyDataSet = path+'/5'
pathNoisyDataSet = path+'/NewDataset/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []
nametrainNoisyDataSet = []

unet = Unet()

ramdomIdx = np.random.randint(1, len(arrayGroundTruth)+1, 2).tolist()
print(ramdomIdx)

nameGroundTruth = []
for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    groundTruthDataSet = sitk.ReadImage(pathGroundTruthElement)
    groundTruthDataSet = sitk.GetArrayFromImage(groundTruthDataSet)
    name, extension = os.path.splitext(element)
    if extension == '.nii':
        name, extension2 = os.path.splitext(name)
    ind = name.find('Subject')
    name = name[ind + len('Subject'):]
    nameGroundTruth.append(name)

    nametrainNoisyDataSet = 'noisyDataSet5_Subject'+name+'.nii'
    pathNoisyDataSetElement = pathNoisyDataSet + '/' + nametrainNoisyDataSet
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

    if int(name) not in ramdomIdx:
        trainGroundTruth.append(groundTruthDataSet)
        trainNoisyDataSet.append(noisyDataSet)
    else:
        validGroundTruth.append(groundTruthDataSet)
        validNoisyDataSet.append(noisyDataSet)

#for element in arrayNoisyDataSet:
#    pathNoisyDataSetElement = pathNoisyDataSet+'/'+element
#    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
#    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
#    name, extension = os.path.splitext(element)
#    if extension == '.nii':
#        name, extension2 = os.path.splitext(name)
#    ind = name.find('Subject')
#    name = name[ind + len('Subject'):]
#    nametrainNoisyDataSet.append(name)
#    if int(name) not in ramdomIdx:
#        trainNoisyDataSet.append(noisyDataSet)
#    else:
#        validNoisyDataSet.append(noisyDataSet)


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


lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer, batchSize, epochs, device, save = True, name = 'Model6')

df = pd.DataFrame(lossValuesTraining)
df.to_excel('lossValuesTrainingSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesEpoch)
df.to_excel('lossValuesTrainingSetEpochModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSet)
df.to_excel('lossValuesDevSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSetAllEpoch)
df.to_excel('lossValuesDevSetEpochModel6Total.xlsx')

import pandas as pd
import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage import rotate
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import math

import torch
import torch.optim as optim

from torch import nn
from utils import trainModel
from utils import reshapeDataSet
from unet import Unet
from utils import saveDataCsv

#from unet import UnetWithResidual5Layers
from unet import UnetWithResidual
#from unet import Unet512
#from unet import UnetDe1a16Hasta512

def lossFunction(output,target):
    loss = (torch.sum((output - target)**2)) / torch.sum(target)
    return loss

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

######################## TRAINING PARAMETERS ###############
batchSize = 8
epochs = 150
learning_rate=0.00005
printStep_epochs = 1
plotStep_epochs = 5
printStep_batches = 100
plotStep_batches = math.inf

lowDose_perc = 5
fullDose = 100

## Type of normalization ##
normMeanStdSlice = True
normMeanSlice = False

nameThisNet = 'UnetResidual_MSE_lr5e-05_AlignTrue_MeanStdSlice'.format(learning_rate)

outputPath = '../../results/' + nameThisNet + '/'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../data/BrainWebSimulationsCompleteDataSet'


# Choosing UNet for training
#unet = Unet(1,1)
unet = UnetWithResidual(1, 1)

pathNoisyDataSet = path+'/'+str(lowDose_perc)
pathGroundTruth = path+'/'+str(fullDose)
pathGreyMatter = path+'/grey mask'
pathWhiteMatter = path+'/white mask'

groundTruth = sitk.ReadImage(pathGroundTruth+'/groundTruthDataSet.nii')
voxelSize_mm = groundTruth.GetSpacing()
groundTruthDataSet = sitk.GetArrayFromImage(groundTruth)

noisyDataSet = sitk.ReadImage(pathNoisyDataSet+'/NoisyDataSet_dose_5.nii')
noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

greyMatter = sitk.ReadImage(pathGreyMatter+'/greyMaskDataSet.nii')
greyMatter = sitk.GetArrayFromImage(greyMatter)

whiteMatter = sitk.ReadImage(pathWhiteMatter+'/whiteMaskDataSet.nii')
whiteMatter = sitk.GetArrayFromImage(whiteMatter)

cantNrosAleatorios = round(len(noisyDataSet) * 0.2)
numerosAleatorios = random.sample(range(0, len(noisyDataSet) + 1), cantNrosAleatorios)

validNoisyDataSet = []
validGroundTruth = []
validGreyMask = []
validWhiteMask = []
trainNoisyDataSet = []
trainGroundTruth = []

if normMeanSlice:
    meanSubjectNoisy = np.mean(np.mean(noisyDataSet, axis=-1), axis=-1)
    meanSubjectGroundTruth = np.mean(np.mean(groundTruthDataSet, axis=-1), axis=-1)

    meanSubjectNoisy = np.where(meanSubjectNoisy == 0, np.nan, meanSubjectNoisy)
    meanSubjectGroundTruth = np.where(meanSubjectGroundTruth == 0, np.nan, meanSubjectGroundTruth)

    subjectNoisyNorm = noisyDataSet / meanSubjectNoisy[:, None, None]
    subjectGroundTruthNorm = groundTruthDataSet /meanSubjectGroundTruth[:, None, None]

    subjectNoisyNorm = np.nan_to_num(subjectNoisyNorm, nan=0)
    subjectGroundTruthNorm = np.nan_to_num(subjectGroundTruthNorm, nan=0)

if normMeanStdSlice:
    meanSubjectNoisy = np.mean(np.mean(noisyDataSet, axis=1), axis=1)
    meanSubjectGroundTruth = np.mean(np.mean(groundTruthDataSet, axis=1), axis=1)

    stdSubjectNoisy = np.std(np.std(noisyDataSet, axis=1), axis=1)
    stdSubjectNoisy = np.where(stdSubjectNoisy == 0, np.nan, stdSubjectNoisy)

    stdSubjectGroundTruth = np.std(np.std(groundTruthDataSet, axis=1), axis=1)
    stdSubjectGroundTruth = np.where(stdSubjectGroundTruth == 0, np.nan, stdSubjectGroundTruth)

    subjectNoisyNorm = (noisyDataSet - meanSubjectNoisy[:, None, None]) / stdSubjectNoisy[:, None, None]
    subjectGroundTruthNorm = (groundTruthDataSet - meanSubjectGroundTruth[:, None, None]) / stdSubjectGroundTruth[:, None, None]

    subjectNoisyNorm = np.nan_to_num(subjectNoisyNorm, nan=0)
    subjectGroundTruthNorm = np.nan_to_num(subjectGroundTruthNorm, nan=0)

if normMeanStdSlice == False and normMeanSlice == False:
    subjectNoisyNorm = noisyDataSet
    subjectGroundTruthNorm = groundTruthDataSet


validNoisyDataSetToSave = []
validGroundTruthToSave = []

for slc in range(0,len(noisyDataSet)):
    if slc in numerosAleatorios:
        validNoisyDataSetToSave.append(noisyDataSet[slc].copy())
        validNoisyDataSet.append(subjectNoisyNorm[slc].copy())
        validGroundTruthToSave.append(groundTruthDataSet[slc].copy())
        validGroundTruth.append(subjectGroundTruthNorm[slc].copy())
        validGreyMask.append(greyMatter[slc].copy())
        validWhiteMask.append(whiteMatter[slc].copy())

    else:
        trainNoisyDataSet.append(subjectNoisyNorm[slc].copy())
        trainGroundTruth.append(subjectGroundTruthNorm[slc].copy())


trainGroundTruth = np.array(trainGroundTruth)
validGroundTruth = np.array(validGroundTruth)
validGroundTruthToSave = np.array(validGroundTruthToSave)

trainNoisyDataSet = np.array(trainNoisyDataSet)
validNoisyDataSet = np.array(validNoisyDataSet)
validNoisyDataSetToSave = np.array(validNoisyDataSetToSave)

image = sitk.GetImageFromArray(np.array(validGroundTruthToSave))
image.SetSpacing(voxelSize_mm)
nameImage = 'validGroundTruth_dose_' + str(lowDose_perc) + '.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(validNoisyDataSetToSave))
image.SetSpacing(voxelSize_mm)
nameImage = 'validNoisyDataset.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(validGreyMask))
image.SetSpacing(voxelSize_mm)
nameImage = 'greyMaskValidSet.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(validWhiteMask))
image.SetSpacing(voxelSize_mm)
nameImage = 'whiteMaskValidSet.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

trainGroundTruthNorm = reshapeDataSet(trainGroundTruth.squeeze())
validGroundTruthNorm = reshapeDataSet(validGroundTruth.squeeze())
trainNoisyDataSetNorm = reshapeDataSet(trainNoisyDataSet.squeeze())
validNoisyDataSetNorm = reshapeDataSet(validNoisyDataSet.squeeze())

# Shuffle the data:
trainNoisyDataSetNorm, trainGroundTruthNorm = shuffle(trainNoisyDataSetNorm, trainGroundTruthNorm, random_state=0)
validNoisyDataSetNorm, validGroundTruthNorm = shuffle(validNoisyDataSetNorm, validGroundTruthNorm, random_state=0)

print('Valid GT:',validGroundTruthNorm.shape)

print('Train Noisy Dataset:',trainNoisyDataSetNorm.shape)
print('Valid Noisy Dataset:',validNoisyDataSetNorm.shape)

# Create dictionaries with training sets:
trainingSet = dict([('input',trainNoisyDataSetNorm), ('output', trainGroundTruthNorm)])
validSet = dict([('input',validNoisyDataSetNorm), ('output', validGroundTruthNorm)])

print('Data set size. Training set: {0}. Valid set: {1}.'.format(trainingSet['input'].shape[0], validSet['input'].shape[0]))

# Entrenamiento #
# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)




lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer, batchSize,
                                                                                            epochs, device, pre_trained = False, save = True, outputPath=outputPath,name = nameThisNet,
                                                                                            printStep_batches = printStep_batches, plotStep_batches = plotStep_batches,
                                                                                            printStep_epochs = printStep_epochs, plotStep_epochs = plotStep_epochs)
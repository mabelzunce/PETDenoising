import nibabel as nb
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy
import numpy as np
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
#from unet import UnetWithResidual5Layers
#from unet import Unet512
#from unet import UnetDe1a16Hasta512

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

######################## TRAINING PARAMETERS ###############
batchSize = 8
epochs = 100
learning_rate=0.00005
printStep_epochs = 1
plotStep_epochs = 5
printStep_batches = 100
plotStep_batches = math.inf

normalizeInput = True
nameThisNet = 'Unet5LayersNewArchitecture_MSE_lr5e-05_AlignTrue'.format(learning_rate)
if normalizeInput:
    nameThisNet = nameThisNet + '_normMeanValue'

outputPath = '../../results/' + nameThisNet + '/'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../data/BrainWebSimulations/'
lowDose_perc = 5
actScaleFactor = 100/lowDose_perc

pathGroundTruth = path+'/100'
#pathGroundTruth = path+'/NewDataset/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

pathNoisyDataSet = path + str(lowDose_perc)
#pathNoisyDataSet = path+'/NewDataset/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []
nametrainNoisyDataSet = []

unet = Unet(1,1)
#unet = Unet512(1,1,32)
#unet = UnetDe1a16Hasta512(1,1,16)
#unet = UnetWithResidual(1, 1)
#unet = UnetWithResidual5Layers(1, 1)

rng = np.random.default_rng()
#ramdomIdx = rng.choice(len(arrayGroundTruth)+1, int(4), replace=False)
#ramdomIdx = np.random.randint(1, len(arrayGroundTruth)+1, 4).tolist()
# Fixed validation phantoms:
ramdomIdx = [2, 4, 6, 8]
print(ramdomIdx)

nameGroundTruth = []
for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    groundTruthDataSet = sitk.ReadImage(pathGroundTruthElement)
    groundTruthDataSet = sitk.GetArrayFromImage(groundTruthDataSet)
    name, extension = os.path.splitext(element)
    if extension == '.gz':
        name, extension2 = os.path.splitext(name)
        extension = extension2 + extension
    ind = name.find('Subject')
    name = name[ind + len('Subject'):]
    nameGroundTruth.append(name)

    nametrainNoisyDataSet = 'noisyDataSet5_Subject'+name+ extension
    pathNoisyDataSetElement = pathNoisyDataSet + '/' + nametrainNoisyDataSet
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

    if int(name) not in ramdomIdx:
        trainGroundTruth.append(groundTruthDataSet)
        trainNoisyDataSet.append(noisyDataSet)
    else:
        validGroundTruth.append(groundTruthDataSet)
        validNoisyDataSet.append(noisyDataSet)

## Set de entramiento
trainNoisyDataSetNorm = []
trainGroundTruthNorm = []
for subject in range(0, len(trainNoisyDataSet)):
    subjectElementNoisy = trainNoisyDataSet[subject]
    subjectElementGroundTruth = trainGroundTruth[subject]
    for slice in range(0, subjectElementNoisy.shape[0]):
        #maxSliceNoisy = subjectElementNoisy[slice, :, :].max()
        #maxSliceGroundTruth = subjectElementGroundTruth[slice, :, :].max()
        meanSliceNoisy = subjectElementNoisy[slice, :, :].mean()
        meanSliceGroundTruth = subjectElementGroundTruth[slice, :, :].mean()
        if (meanSliceNoisy > 0.0000001) and (meanSliceGroundTruth > 0.0) :
            normNoisy = ((subjectElementNoisy[slice, :, :]) )*actScaleFactor # This factor scales up the activity to the full dose values
            normGroundTruth = ((subjectElementGroundTruth[slice, :, :]))
            meanSliceNoisy = normNoisy.mean()
            meanSliceGroundTruth = normGroundTruth.mean()
            if normalizeInput:
                normNoisy = normNoisy / meanSliceNoisy
                normGroundTruth = normGroundTruth / meanSliceNoisy # normalize by the input not by the groundtruth, maxSliceGroundTruth
            trainNoisyDataSetNorm.append(normNoisy)
            trainNoisyDataSetNorm.append(np.rot90(normNoisy))
            trainNoisyDataSetNorm.append(rotate(normNoisy, angle=45, reshape=False))
            trainGroundTruthNorm.append(normGroundTruth )
            trainGroundTruthNorm.append(np.rot90(normGroundTruth))
            trainGroundTruthNorm.append(rotate(normGroundTruth, angle=45, reshape=False))

# Set de validacion
validNoisyDataSetNorm = []
validGroundTruthNorm = []
for subject in range(0, len(validNoisyDataSet)):
    subjectElementNoisy = validNoisyDataSet[subject]
    subjectElementGroundTruth = validGroundTruth[subject]
    for slice in range(0, subjectElementNoisy.shape[0]):
        #maxSliceNoisy = subjectElementNoisy[slice, :, :].max()
        #maxSliceGroundTruth = subjectElementGroundTruth[slice, :, :].max()
        meanSliceNoisy = subjectElementNoisy[slice, :, :].mean()
        meanSliceGroundTruth = subjectElementGroundTruth[slice, :, :].mean()
        if (meanSliceNoisy > 0.0000001) and (meanSliceGroundTruth > 0.0):
            normNoisy = ((subjectElementNoisy[slice, :, :]) )*actScaleFactor # This factor scales up the activity to the full dose values
            normGroundTruth = ((subjectElementGroundTruth[slice, :, :]))
            meanSliceNoisy = normNoisy.mean()
            meanSliceGroundTruth = normGroundTruth.mean()
            if normalizeInput:
                normNoisy = normNoisy / meanSliceNoisy
                normGroundTruth = normGroundTruth / meanSliceNoisy
            validNoisyDataSetNorm.append(normNoisy)
            validNoisyDataSetNorm.append(np.rot90(normNoisy))
            validNoisyDataSetNorm.append(rotate(normNoisy, angle=45, reshape=False))
            validGroundTruthNorm.append(normGroundTruth)
            validGroundTruthNorm.append(np.rot90(normGroundTruth))
            validGroundTruthNorm.append(rotate(normGroundTruth, angle=45, reshape=False))

trainGroundTruthNorm = np.array(trainGroundTruthNorm)
validGroundTruthNorm = np.array(validGroundTruthNorm)
trainNoisyDataSetNorm = np.array(trainNoisyDataSetNorm)
validNoisyDataSetNorm = np.array(validNoisyDataSetNorm)

trainGroundTruthNorm = reshapeDataSet(trainGroundTruthNorm)
validGroundTruthNorm = reshapeDataSet(validGroundTruthNorm)
trainNoisyDataSetNorm = reshapeDataSet(trainNoisyDataSetNorm)
validNoisyDataSetNorm = reshapeDataSet(validNoisyDataSetNorm)

# Shuffle the data:
trainNoisyDataSetNorm, trainGroundTruthNorm = shuffle(trainNoisyDataSetNorm, trainGroundTruthNorm, random_state=0)
validNoisyDataSetNorm, validGroundTruthNorm = shuffle(validNoisyDataSetNorm, validGroundTruthNorm, random_state=0)

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
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)


lossValuesTraining,lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch = trainModel(unet,trainingSet, validSet,criterion,optimizer, batchSize,
                                                                                            epochs, device, pre_trained = False, save = True, outputPath=outputPath,name = nameThisNet,
                                                                                            printStep_batches = printStep_batches, plotStep_batches = plotStep_batches,
                                                                                            printStep_epochs = printStep_epochs, plotStep_epochs = plotStep_epochs)

df = pd.DataFrame(lossValuesTraining)
df.to_excel('lossValuesTrainingSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesEpoch)
df.to_excel('lossValuesTrainingSetEpochModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSet)
df.to_excel('lossValuesDevSetBatchModel6Total.xlsx')

df = pd.DataFrame(lossValuesDevSetAllEpoch)
df.to_excel('lossValuesDevSetEpochModel6Total.xlsx')
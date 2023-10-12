import pandas as pd
import SimpleITK as sitk
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
epochs = 100
learning_rate=0.00005
printStep_epochs = 1
plotStep_epochs = 5
printStep_batches = 100
plotStep_batches = math.inf

normalizeInput = False
normalizeInputMeanGlobal = False
normalizeInputMeanGlobalWithoutZeros = False
normalizeInputMeanSlicelWithoutZeros = True
normalizeInputMaxGlobal = False
normalizeInputMeanStdSlice = False
normalizeInputMeanSlice = False
normalizeInputMaxSlice = False

nameThisNet = 'Unet_MSE_lr5e-05_AlignTrue_MeanSliceWithoutZeros_2023'.format(learning_rate)

outputPath = '../../results/' + nameThisNet + '/'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../data/BrainWebSimulations/'
lowDose_perc = 5
actScaleFactor = 100/lowDose_perc
applyScaleFactor = False

pathGroundTruth = path+'/100'
arrayGroundTruth = os.listdir(pathGroundTruth)
trainGroundTruth = []
validGroundTruth = []

pathNoisyDataSet = path + str(lowDose_perc)
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)
trainNoisyDataSet = []
validNoisyDataSet = []
nametrainNoisyDataSet = []

# Choosing UNet for training
unet = Unet(1,1)
#unet = UnetWithResidual(1, 1)

rng = np.random.default_rng()
# Fixed validation phantoms:
ramdomIdx = [2, 4, 6, 8]
print(ramdomIdx)

validName = []
trainingName = []
nameGroundTruth = []
cantidadSlices = []

for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    groundTruthDataSet = sitk.ReadImage(pathGroundTruthElement)
    voxelSize_mm = groundTruthDataSet.GetSpacing()
    groundTruthDataSet = sitk.GetArrayFromImage(groundTruthDataSet)
    cantidadSlices.append(len(groundTruthDataSet))
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
        trainingName.append(name)
        trainGroundTruth.append(groundTruthDataSet)
        if applyScaleFactor == True:
            trainNoisyDataSet.append((noisyDataSet * actScaleFactor))
        else:
            trainNoisyDataSet.append(noisyDataSet)
    else:
        validName.append(name)
        validGroundTruth.append(groundTruthDataSet)
        if applyScaleFactor == True:
            validNoisyDataSet.append((noisyDataSet * actScaleFactor))
        else:
            validNoisyDataSet.append(noisyDataSet)

## Set de entramiento
trainNoisyDataSetNorm = []
trainGroundTruthNorm = []

subjectNoisyNorm = []
subjectGroundTruthNorm = []

subjectDiezGroundTruth = []
subjectDiezNoisy = []

# kinds of Normalization
for subject in range(0, len(trainNoisyDataSet)):
    subjectElementNoisy = trainNoisyDataSet[subject]
    subjectElementGroundTruth = trainGroundTruth[subject]

    if normalizeInputMaxSlice == True:
        maxSubjectNoisy = np.max(np.max(subjectElementNoisy, axis = 1),axis=1)
        maxSubjectGroundTruth = np.max(np.max(subjectElementGroundTruth, axis = 1),axis=1)

        subjectNoisyNorm = subjectElementNoisy/maxSubjectNoisy[:,None,None]
        subjectGroundTruthNorm = subjectElementGroundTruth / maxSubjectGroundTruth[:,None,None]

    if normalizeInputMeanGlobal == True:
        meanSubjectNoisy = np.mean(subjectElementNoisy)
        meanSubjectGroundTruth = np.mean(subjectElementGroundTruth)

        subjectNoisyNorm = subjectElementNoisy/meanSubjectNoisy
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth #CHEQUEAR

    if normalizeInputMeanGlobalWithoutZeros == True:
        X = np.ma.masked_equal(subjectElementNoisy, 0)
        nonZeros = np.sum((~(X.mask)))
        pixelNonZeros = np.sum(noisyDataSet)
        meanSubjectNoisy = pixelNonZeros / nonZeros

        X = np.ma.masked_equal(subjectElementGroundTruth, 0)
        nonZeros = np.sum((~(X.mask)))
        pixelNonZeros = np.sum(noisyDataSet)
        meanSubjectGroundTruth = pixelNonZeros / nonZeros

        subjectNoisyNorm = (subjectElementNoisy / meanSubjectNoisy)
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth  # CHEQUEAR

    if normalizeInputMeanSlice == True:
        meanSubjectNoisy = np.mean(np.mean(subjectElementNoisy, axis = 1),axis=1)
        meanSubjectGroundTruth = np.mean(np.mean(subjectElementGroundTruth, axis = 1),axis=1)

        subjectNoisyNorm = subjectElementNoisy/meanSubjectNoisy[:,None,None]
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth[:,None,None]

    if normalizeInputMaxGlobal == True:
        maxSubjectNoisy = np.max(subjectElementNoisy)
        maxSubjectGroundTruth = np.max(subjectElementGroundTruth)

        subjectNoisyNorm = subjectElementNoisy / maxSubjectNoisy
        subjectGroundTruthNorm = subjectElementGroundTruth / maxSubjectGroundTruth  # CHEQUEAR

    if normalizeInputMeanStdSlice == True:
        meanSubjectNoisy = np.mean(np.mean(subjectElementNoisy, axis=1), axis=1)
        meanSubjectGroundTruth = np.mean(np.mean(subjectElementGroundTruth, axis=1), axis=1)

        stdSubjectNoisy = np.std(np.std(subjectElementNoisy, axis=1), axis=1)
        stdSubjectGroundTruth = np.std(np.std(subjectElementGroundTruth, axis=1), axis=1)

        subjectNoisyNorm = (subjectElementNoisy - meanSubjectNoisy[:,None,None]) / stdSubjectNoisy[:, None, None]
        subjectGroundTruthNorm = (subjectElementGroundTruth - meanSubjectGroundTruth[:, None, None]) / stdSubjectGroundTruth[:, None, None]

    if applyScaleFactor == True:
        subjectElementNoisy = np.array(subjectElementNoisy)

    # calculo el valor medio de las imagenes para que no influya negativamente en el entrenamiento
    # YA NORMALIZADO
    for slice in range(0, subjectElementNoisy.shape[0]):

        if normalizeInputMeanSlicelWithoutZeros:
            meanSliceNoisy = np.mean(subjectElementNoisy[slice, :, :])
            meanSliceGroundTruth = np.mean(subjectElementGroundTruth[slice, :, :])

        else:
            meanSliceNoisy = np.mean(subjectNoisyNorm[slice,:,:])
            meanSliceGroundTruth = np.mean(subjectGroundTruthNorm[slice,:,:])

        if (meanSliceNoisy > 0.0000001) and (meanSliceGroundTruth > 0.0):

            if normalizeInputMeanSlicelWithoutZeros == True:
                X = np.ma.masked_equal(subjectElementNoisy[slice, :, :], 0)
                nonZeros = np.sum((~(X.mask)))
                pixelNonZeros = np.sum(subjectElementNoisy[slice, :, :])
                meanSubjectNoisy = pixelNonZeros / nonZeros
                #meanSubjectNoisy = np.mean(X)

                X = np.ma.masked_equal(subjectElementGroundTruth[slice, :, :], 0)
                nonZeros = np.sum((~(X.mask)))
                pixelNonZeros = np.sum(subjectElementGroundTruth[slice, :, :])
                meanSubjectGroundTruth = pixelNonZeros / nonZeros

                #meanSubjectGroundTruth = np.mean(X)

                sliceNoisyNorm = subjectElementNoisy[slice, :, :] / meanSubjectNoisy

                sliceGroundTruthNorm = subjectElementGroundTruth[slice, :, :] / meanSubjectGroundTruth  # CHEQUEAR


            else:
                sliceNoisyNorm = subjectNoisyNorm[slice, :, :]
                sliceGroundTruthNorm = subjectGroundTruthNorm[slice, :, :]

            trainNoisyDataSetNorm.append(sliceNoisyNorm.copy())
            trainNoisyDataSetNorm.append(np.rot90(sliceNoisyNorm.copy()))
            trainNoisyDataSetNorm.append(rotate(sliceNoisyNorm.copy(), angle=45, reshape=False))
            trainGroundTruthNorm.append(sliceGroundTruthNorm.copy())
            trainGroundTruthNorm.append(np.rot90(sliceGroundTruthNorm.copy()))
            trainGroundTruthNorm.append(rotate(sliceGroundTruthNorm.copy(), angle=45, reshape=False))

subjectDosGroundTruth = []
subjectDosNoisy = []

# Set de validacion
validNoisyDataSetNorm = []
validGroundTruthNorm = []

# relacion
meanNoisyValuesAntes = []
meanNoisyValuesDespues = []
meanGtValuesAntes = []
meanGtValuesDespues = []


for subject in range(0, len(validNoisyDataSet)):
    subjectElementNoisy = validNoisyDataSet[subject]
    subjectElementGroundTruth = validGroundTruth[subject]

    if normalizeInputMeanSlice == True:
        meanSubjectNoisy = np.mean(np.mean(subjectElementNoisy, axis=1), axis=1)
        meanSubjectGroundTruth = np.mean(np.mean(subjectElementGroundTruth, axis=1), axis=1)

        subjectNoisyNorm = subjectElementNoisy / meanSubjectNoisy[:, None, None]
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth[:, None, None]

    if normalizeInputMaxSlice == True:
        maxSubjectNoisy = np.max(np.max(subjectElementNoisy, axis=1), axis=1)
        maxSubjectGroundTruth = np.max(np.max(subjectElementGroundTruth, axis=1), axis=1)

        subjectNoisyNorm = subjectElementNoisy / maxSubjectNoisy[:, None, None]
        subjectGroundTruthNorm = subjectElementGroundTruth / maxSubjectGroundTruth[:, None, None]

    if normalizeInputMeanGlobal == True:
        meanSubjectNoisy = np.mean(subjectElementNoisy)
        meanSubjectGroundTruth = np.mean(subjectElementGroundTruth)

        subjectNoisyNorm = subjectElementNoisy / meanSubjectNoisy
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth  # CHEQUEAR

    if normalizeInputMeanGlobalWithoutZeros == True:
        X = np.ma.masked_equal(subjectElementNoisy, 0)
        nonZeros = np.sum((~(X.mask)))
        pixelNonZeros = np.sum(noisyDataSet)
        meanSubjectNoisy = pixelNonZeros / nonZeros

        X = np.ma.masked_equal(subjectElementGroundTruth, 0)
        nonZeros = np.sum((~(X.mask)))
        pixelNonZeros = np.sum(noisyDataSet)
        meanSubjectGroundTruth = pixelNonZeros / nonZeros

        subjectNoisyNorm = (subjectElementNoisy / meanSubjectNoisy)
        subjectGroundTruthNorm = subjectElementGroundTruth / meanSubjectGroundTruth  # CHEQUEAR

    if normalizeInputMaxGlobal == True:
        maxSubjectNoisy = np.max(subjectElementNoisy)
        maxSubjectGroundTruth = np.max(subjectElementGroundTruth)

        subjectNoisyNorm = subjectElementNoisy / maxSubjectNoisy
        subjectGroundTruthNorm = subjectElementGroundTruth / maxSubjectGroundTruth  # CHEQUEAR

    if normalizeInputMeanStdSlice == True:
        meanSubjectNoisy = np.mean(np.mean(subjectElementNoisy, axis=1), axis=1)
        meanSubjectGroundTruth = np.mean(np.mean(subjectElementGroundTruth, axis=1), axis=1)

        stdSubjectNoisy = np.std(np.std(subjectElementNoisy, axis=1), axis=1)
        stdSubjectGroundTruth = np.std(np.std(subjectElementGroundTruth, axis=1), axis=1)

        subjectNoisyNorm = (subjectElementNoisy - meanSubjectNoisy[:,None,None]) / stdSubjectNoisy[:, None, None]
        subjectGroundTruthNorm = (subjectElementGroundTruth - meanSubjectGroundTruth[:, None, None]) / stdSubjectGroundTruth[:, None, None]


    # calculo el valor medio de las imagenes para que no influya negativamente en el entrenamiento
    # YA NORMALIZADO
    for slice in range(0, subjectElementNoisy.shape[0]):

        if normalizeInputMeanSlicelWithoutZeros:
            meanSliceNoisy = np.mean(subjectElementNoisy[slice, :, :])
            meanSliceGroundTruth = np.mean(subjectElementGroundTruth[slice, :, :])

        else:
            meanSliceNoisy = np.mean(subjectNoisyNorm[slice,:,:])
            meanSliceGroundTruth = np.mean(subjectGroundTruthNorm[slice,:,:])

        if (meanSliceNoisy > 0.0000001) and (meanSliceGroundTruth > 0.0):

            if normalizeInputMeanSlicelWithoutZeros == True:
                X = np.ma.masked_equal(subjectElementNoisy[slice, :, :], 0)

                nonZeros = np.sum((~(X.mask)))
                pixelNonZeros = np.sum(subjectElementNoisy[slice, :, :])
                meanSubjectNoisy = pixelNonZeros / nonZeros
                #meanSubjectNoisy = np.mean(X)

                X = np.ma.masked_equal(subjectElementGroundTruth[slice, :, :], 0)
                nonZeros = np.sum((~(X.mask)))
                pixelNonZeros = np.sum(subjectElementGroundTruth[slice, :, :])
                meanSubjectGroundTruth = pixelNonZeros / nonZeros

                sliceNoisyNorm = subjectElementNoisy[slice, :, :] / meanSubjectNoisy
                sliceGroundTruthNorm = subjectElementGroundTruth[slice, :, :] / meanSubjectGroundTruth  # CHEQUEAR

            else:
                sliceNoisyNorm = subjectNoisyNorm[slice, :, :]
                sliceGroundTruthNorm = subjectGroundTruthNorm[slice, :, :]

            validNoisyDataSetNorm.append(sliceNoisyNorm.copy())
            validNoisyDataSetNorm.append(np.rot90(sliceNoisyNorm.copy()))
            validNoisyDataSetNorm.append(rotate(sliceNoisyNorm.copy(), angle=45, reshape=False))
            validGroundTruthNorm.append(sliceGroundTruthNorm.copy())
            validGroundTruthNorm.append(np.rot90(sliceGroundTruthNorm.copy()))
            validGroundTruthNorm.append(rotate(sliceGroundTruthNorm.copy(), angle=45, reshape=False))


#saveDataCsv(divisionSlices, 'divisionSlices_MaxValue.csv', pathSaveResults)
saveDataCsv(np.array(meanNoisyValuesAntes), 'meanNoisySlicesAntesNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanNoisyValuesDespues), 'meanNoisySlicesDspNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanGtValuesAntes), 'meanGtSlicesAntesNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanGtValuesDespues), 'meanGtSlicesDspNormGlobal_MeanValue.csv', outputPath)

trainGroundTruthNorm = np.array(trainGroundTruthNorm)
validGroundTruthNorm = np.array(validGroundTruthNorm)
trainNoisyDataSetNorm = np.array(trainNoisyDataSetNorm)
validNoisyDataSetNorm = np.array(validNoisyDataSetNorm)

trainGroundTruthNorm = reshapeDataSet(trainGroundTruthNorm.squeeze())
validGroundTruthNorm = reshapeDataSet(validGroundTruthNorm.squeeze())
trainNoisyDataSetNorm = reshapeDataSet(trainNoisyDataSetNorm.squeeze())
validNoisyDataSetNorm = reshapeDataSet(validNoisyDataSetNorm.squeeze())

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
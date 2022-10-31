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
from utils import saveDataCsv

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
nameThisNet = 'Unet5LayersNewArchitecture_MSE_lr5e-05_AlignTrue_GlobalMeanNorm'.format(learning_rate)
if normalizeInput:
    nameThisNet = nameThisNet + '_normMeanValue'

outputPath = '../../results/' + nameThisNet + '/'

#pathSaveResults = 'C:/Users/Encargado/Desktop/'

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
        trainNoisyDataSet.append(noisyDataSet)
    else:
        validName.append(name)
        validGroundTruth.append(groundTruthDataSet)
        validNoisyDataSet.append(noisyDataSet)

## Set de entramiento
trainNoisyDataSetNorm = []
trainGroundTruthNorm = []

subjectDiezGroundTruth = []
subjectDiezNoisy = []

for subject in range(0, len(trainNoisyDataSet)):
    subjectElementNoisy = trainNoisyDataSet[subject]
    X = np.ma.masked_equal(subjectElementNoisy, 0)
    nonZeros = np.sum((~(X.mask)))
    pixelNonZeros = np.sum(subjectElementNoisy)
    meanSubjectNoisy = pixelNonZeros / nonZeros

    #meanSubjectNoisy = subjectElementNoisy.max()

    subjectElementGroundTruth = trainGroundTruth[subject]

    #X = np.ma.masked_equal(subjectElementGroundTruth, 0)
    #nonZeros = np.sum((~(X.mask)))
    #pixelNonZeros = np.sum(subjectElementGroundTruth)
    #meanSubjectGroundTruth = pixelNonZeros / nonZeros

    if normalizeInput:
        subjectNoisyMeanNorm = subjectElementNoisy/meanSubjectNoisy
        subjectGroundTruthMeanNorm = subjectElementGroundTruth / meanSubjectNoisy

    if int(trainingName[subject]) == 10:
        subjectDiezGroundTruth.append(subjectGroundTruthMeanNorm)
        subjectDiezNoisy.append(subjectNoisyMeanNorm)

    for slice in range(0, subjectElementNoisy.shape[0]):
        meanSliceNoisy = subjectNoisyMeanNorm[slice, :, :].mean()
        meanSliceGroundTruth = subjectGroundTruthMeanNorm[slice, :, :].mean()

        if (meanSliceNoisy > 0.0000001) and (meanSliceGroundTruth > 0.0):
            sliceNoisyNorm = subjectNoisyMeanNorm[slice, :, :]
            sliceGroundTruthNorm = subjectGroundTruthMeanNorm[slice, :, :]

            trainNoisyDataSetNorm.append(sliceNoisyNorm)
            trainNoisyDataSetNorm.append(np.rot90(sliceNoisyNorm))
            trainNoisyDataSetNorm.append(rotate(sliceNoisyNorm, angle=45, reshape=False))
            trainGroundTruthNorm.append(sliceGroundTruthNorm)
            trainGroundTruthNorm.append(np.rot90(sliceGroundTruthNorm))
            trainGroundTruthNorm.append(rotate(sliceGroundTruthNorm, angle=45, reshape=False))

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
    X = np.ma.masked_equal(subjectElementNoisy, 0)
    nonZeros = np.sum((~(X.mask)))
    pixelNonZeros = np.sum(subjectElementNoisy)
    meanSubjectNoisy = pixelNonZeros/nonZeros

    #meanSubjectNoisy = subjectElementNoisy.max()

    subjectElementGroundTruth = validGroundTruth[subject]

    #X = np.ma.masked_equal(subjectElementGroundTruth, 0)
    #nonZeros = np.sum((~(X.mask)))
    #pixelNonZeros = np.sum(subjectElementGroundTruth)
    #meanSubjectGroundTruth = pixelNonZeros / nonZeros

    if normalizeInput:
        normNoisy = subjectElementNoisy / meanSubjectNoisy
        normGroundTruth = subjectElementGroundTruth / meanSubjectNoisy

    if int(validName[subject]) == 2:
        subjectDosGroundTruth.append(normGroundTruth)
        subjectDosNoisy.append(normNoisy)

    for slice in range(0, subjectElementNoisy.shape[0]):
        meanSliceNoisyNorm = normNoisy[slice, :, :].mean()
        meanSliceGroundTruthNorm = normGroundTruth[slice, :, :].mean()

        if (meanSliceNoisyNorm > 0.0000001) and (meanSliceGroundTruthNorm > 0.0):
            sliceNoisyNorm = normNoisy[slice, :, :]
            sliceGroundTruthNorm = normGroundTruth[slice, :, :]

            validNoisyDataSetNorm.append(sliceNoisyNorm)
            validNoisyDataSetNorm.append(np.rot90(sliceNoisyNorm))
            validNoisyDataSetNorm.append(rotate(sliceNoisyNorm, angle=45, reshape=False))
            validGroundTruthNorm.append(sliceGroundTruthNorm)
            validGroundTruthNorm.append(np.rot90(sliceGroundTruthNorm))
            validGroundTruthNorm.append(rotate(sliceGroundTruthNorm, angle=45, reshape=False))

            if int(validName[subject]) == 2:
                meanNoisyValuesAntes.append(subjectElementNoisy[slice, :, :].mean())
                meanGtValuesAntes.append(subjectElementGroundTruth[slice, :, :].mean())

                meanNoisyValuesDespues.append(normNoisy[slice, :, :].mean())
                meanGtValuesDespues.append(normGroundTruth[slice, :, :].mean())

#divisionSlices = np.array(subjectDosNoisy)/np.array(subjectDosGroundTruth)
#divisionSlices = np.nan_to_num(divisionSlices)
#divisionSlices = divisionSlices.mean(axis=2).mean(axis=1)

#plt.figure(1)
#plt.imshow(divisionSlices[50])

#plt.figure(2)
#plt.imshow(subjectDosGroundTruth[50])

#plt.figure(3)
#plt.imshow(subjectDosNoisy[50])


#saveDataCsv(divisionSlices, 'divisionSlices_MaxValue.csv', pathSaveResults)
saveDataCsv(np.array(meanNoisyValuesAntes), 'meanNoisySlicesAntesNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanNoisyValuesDespues), 'meanNoisySlicesDspNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanGtValuesAntes), 'meanGtSlicesAntesNormGlobal_MeanValue.csv', outputPath)
saveDataCsv(np.array(meanGtValuesDespues), 'meanGtSlicesDspNormGlobal_MeanValue.csv', outputPath)


image = sitk.GetImageFromArray(np.array(subjectDosGroundTruth).squeeze())
image.SetSpacing(voxelSize_mm)
nameImage = 'ValidGroundTruthMeanGlobal_Subject2.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(subjectDosNoisy).squeeze())
image.SetSpacing(voxelSize_mm)
nameImage = 'ValidNoisyMeanGlobal_Subject2.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(subjectDiezGroundTruth).squeeze())
image.SetSpacing(voxelSize_mm)
nameImage = 'TrainGroundTruthMeanGlobal_Subject10.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

image = sitk.GetImageFromArray(np.array(subjectDiezNoisy).squeeze())
image.SetSpacing(voxelSize_mm)
nameImage = 'TrainNoisyMeanGlobal_Subject10.nii'
save_path = os.path.join(outputPath, nameImage)
sitk.WriteImage(image, save_path)

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
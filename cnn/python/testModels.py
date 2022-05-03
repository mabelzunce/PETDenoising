import torch
import torchvision

from unetM import Unet
from utils import imshow
from utils import reshapeDataSet
from utils import MSE
from utils import torchToNp
from utils import mseAntDspModelTorchSlice
from utils import testModelSlice
from utils import obtenerMask
from utils import showGridNumpyImg
from utils import saveNumpyAsNii
from utils import getTestOneModelOneSlices

import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


modelDT1 = Unet()
modelDT1.load_state_dict(torch.load('bestModelDataSet1_24'))

modelDT2 = Unet()
modelDT2.load_state_dict(torch.load('bestModelDataSet2_17'))

noisyDataSet1_nii = sitk.ReadImage('./noisyDataSet1.nii')
img_noisyDataSet1 = sitk.GetArrayFromImage(noisyDataSet1_nii)

noisyDataSet2_nii = sitk.ReadImage('./noisyDataSet2.nii')
img_noisyDataSet2 = sitk.GetArrayFromImage(noisyDataSet2_nii)

groundTruth_nii = sitk.ReadImage('./groundTruth.nii')
img_groundTruth = sitk.GetArrayFromImage(groundTruth_nii)

img_noisyDataSet1 = reshapeDataSet(img_noisyDataSet1)
img_noisyDataSet2 = reshapeDataSet(img_noisyDataSet2)
img_groundTruth = reshapeDataSet(img_groundTruth)

trainingSet1 = dict([('input',img_noisyDataSet1), ('output', img_groundTruth)])
inputsTestSet1 = torch.from_numpy(trainingSet1['input'][:,:,:,:])
groundTruthTestSet1 = torch.from_numpy(trainingSet1['output'][:,:,:,:])


trainingSet2 = dict([('input',img_noisyDataSet2), ('output', img_groundTruth)])
inputsTestSet2 = torch.from_numpy(trainingSet2['input'][:,:,:,:])
groundTruthTestSet2 = torch.from_numpy(trainingSet2['output'][:,:,:,:])



################################### MODELO 1 ##############################################
######################## Testeo data set 1  ###############################

print('-------------------------------------------------')
print('------------MODEL 1------------------------------')
print('-------------------------------------------------')

#inputsTestSet1Rotado = torchvision.transforms.functional.rotate(inputsTestSet1,15)
#groundTruthTestSet1Rotado = torchvision.transforms.functional.rotate(groundTruthTestSet1,15)

print('Data set 1')

getTestOneModelOneSlices(modelDT1,inputsTestSet1[30], groundTruthTestSet1[30], mse = 'True')
getTestOneModelOneSlices(modelDT1,inputsTestSet1[30], groundTruthTestSet1[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT1,inputsTestSet1[30], groundTruthTestSet1[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes
cantImg = 3
cantIdx = (inputsTestSet1.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet1Np = []
outMode1Np = []
groundTruthTestSet1Np = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []


for idx in ramdomIdx :
    inputsTestSet1Np.append(torchToNp(inputsTestSet1[idx]))
    outModel = testModelSlice(modelDT1,inputsTestSet1[idx])
    outModel = torchToNp(outModel)
    outMode1Np.append(outModel[0,:,:,:])
    groundTruthTestSet1Np.append(torchToNp(groundTruthTestSet1[idx]))

    out,antesGrey,dpsGrey = getTestOneModelOneSlices(modelDT1, inputsTestSet1[idx], groundTruthTestSet1[idx], mseGrey='True', greyMatterValue=8)
    out,antesWhite,dpsWhite = getTestOneModelOneSlices(modelDT1, inputsTestSet1[idx], groundTruthTestSet1[idx], mseWhite='True', whiteMatterValue=2)

    mseAntesGrey.append(antesGrey)
    mseDspGrey.append(dpsGrey)

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)


showGridNumpyImg(inputsTestSet1Np,outMode1Np,groundTruthTestSet1Np, plotTitle = 'Modelo1+DATASET1')

## EXCEL
df = pd.DataFrame()

modelo = np.ones([cantImg * 1])

df['Modelo'] = modelo
df['Slice Nro'] = ramdomIdx
df['MSE Grey antes'] = mseAntesGrey
df['MSE Grey dsp'] = mseDspGrey

df['MSE White antes'] = mseAntesWhite
df['MSE White dsp'] = mseDspWhite

df.to_excel('MODELO1+DATASET1.xlsx')

print('Data set 1 NORMALIZADO')

meanGroundTruth = np.mean(torchToNp(groundTruthTestSet1[30]))
meanInputs = np.mean(torchToNp(inputsTestSet1[30]))

newNormImage = (inputsTestSet1[30]/meanInputs)*meanGroundTruth

getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet1[30], mse = 'True')
getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet1[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet1[30], mseWhite = 'True', whiteMatterValue= 2)


## Imagenes NORMALIZADAS
cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

for idx in ramdomIdx:
    meanGroundTruth = np.mean(torchToNp(groundTruthTestSet1[idx]))
    meanInputs = np.mean(torchToNp(inputsTestSet1[idx]))
    newNormImage = (inputsTestSet1[idx] / meanInputs) * meanGroundTruth

    inputsTestSet2Np.append(torchToNp(newNormImage))
    outModel = testModelSlice(modelDT1, newNormImage)
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0, :, :, :])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet2[idx]))

showGridNumpyImg(inputsTestSet2Np, outMode2Np, groundTruthTestSet2Np, plotTitle='Modelo1+DATASET1+Norm')

######################## Testeo data set 2 ###############################

print('Data set 2')

getTestOneModelOneSlices(modelDT1,inputsTestSet2[30], groundTruthTestSet2[30], mse = 'True')
getTestOneModelOneSlices(modelDT1,inputsTestSet2[30], groundTruthTestSet2[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT1,inputsTestSet2[30], groundTruthTestSet2[30], mseWhite = 'True', whiteMatterValue= 2)

cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []

for idx in ramdomIdx :
    inputsTestSet2Np.append(torchToNp(inputsTestSet2[idx]))
    outModel = testModelSlice(modelDT1,inputsTestSet2[idx])
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0,:,:,:])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet2[idx]))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(modelDT1, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                       mseGrey='True', greyMatterValue=8)
    out, antesWhite, dpsWhite = getTestOneModelOneSlices(modelDT1, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                         mseWhite='True', whiteMatterValue=2)

    mseAntesGrey.append(antesGrey)
    mseDspGrey.append(dpsGrey)

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)

## EXCEL
df = pd.DataFrame()

modelo = np.ones([cantImg * 1])

df['Modelo'] = modelo
df['Slice Nro'] = ramdomIdx
df['MSE Grey antes'] = mseAntesGrey
df['MSE Grey dsp'] = mseDspGrey

df['MSE White antes'] = mseAntesWhite
df['MSE White dsp'] = mseDspWhite

df.to_excel('MODELO1+DATASET2.xlsx')


showGridNumpyImg(inputsTestSet2Np,outMode2Np,groundTruthTestSet2Np, plotTitle = 'Modelo1+DATASET2')

print('Data set 2 NORMALIZADO')

meanGroundTruth = np.mean(torchToNp(groundTruthTestSet2[30]))
meanInputs = np.mean(torchToNp(inputsTestSet2[30]))

newNormImage = (inputsTestSet2[30]/meanInputs)*meanGroundTruth

getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet2[30], mse = 'True')
getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet2[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT1,newNormImage, groundTruthTestSet2[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes NORMALIZADAS
cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

for idx in ramdomIdx:
    meanGroundTruth = np.mean(torchToNp(groundTruthTestSet2[idx]))
    meanInputs = np.mean(torchToNp(inputsTestSet2[idx]))
    newNormImage = (inputsTestSet2[idx] / meanInputs) * meanGroundTruth

    inputsTestSet2Np.append(torchToNp(newNormImage))
    outModel = testModelSlice(modelDT1, newNormImage)
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0, :, :, :])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet2[idx]))

showGridNumpyImg(inputsTestSet2Np, outMode2Np, groundTruthTestSet2Np, plotTitle='Modelo1+DATASET2+Norm')




################################### MODELO 2 ##############################################
######################## Testeo data set 1 ###############################
print('-------------------------------------------------')
print('------------MODEL 2------------------------------')
print('-------------------------------------------------')


print('Data set 1')

getTestOneModelOneSlices(modelDT2,inputsTestSet1[30], groundTruthTestSet1[30], mse = 'True')
getTestOneModelOneSlices(modelDT2,inputsTestSet1[30], groundTruthTestSet1[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT2,inputsTestSet1[30], groundTruthTestSet1[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes
cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []

for idx in ramdomIdx :
    inputsTestSet2Np.append(torchToNp(inputsTestSet1[idx]))
    outModel = testModelSlice(modelDT2,inputsTestSet1[idx])
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0,:,:,:])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet1[idx]))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(modelDT2, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                       mseGrey='True', greyMatterValue=8)
    out, antesWhite, dpsWhite = getTestOneModelOneSlices(modelDT2, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                         mseWhite='True', whiteMatterValue=2)

    mseAntesGrey.append(antesGrey)
    mseDspGrey.append(dpsGrey)

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)

showGridNumpyImg(inputsTestSet2Np,outMode2Np,groundTruthTestSet2Np, plotTitle = 'Modelo2+DATASET1')



## EXCEL
df = pd.DataFrame()

modelo = np.ones([cantImg * 1])

df['Modelo'] = modelo
df['Slice Nro'] = ramdomIdx
df['MSE Grey antes'] = mseAntesGrey
df['MSE Grey dsp'] = mseDspGrey

df['MSE White antes'] = mseAntesWhite
df['MSE White dsp'] = mseDspWhite

df.to_excel('MODELO2+DATASET1.xlsx')
print('Data set 1 NORMALIZADO')

meanGroundTruth = np.mean(torchToNp(groundTruthTestSet1[30]))
meanInputs = np.mean(torchToNp(inputsTestSet1[30]))

newNormImage = (inputsTestSet1[30]/meanInputs)*meanGroundTruth

getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet1[30], mse = 'True')
getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet1[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet1[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes NORMALIZADAS
cantImg = 3
cantIdx = (inputsTestSet1.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet1Np = []
outMode1Np = []
groundTruthTestSet1Np = []

for idx in ramdomIdx:
    meanGroundTruth = np.mean(torchToNp(groundTruthTestSet1[idx]))
    meanInputs = np.mean(torchToNp(inputsTestSet1[idx]))
    newNormImage = (inputsTestSet1[idx] / meanInputs) * meanGroundTruth

    inputsTestSet1Np.append(torchToNp(newNormImage))
    outModel = testModelSlice(modelDT2, newNormImage)
    outModel = torchToNp(outModel)
    outMode1Np.append(outModel[0, :, :, :])
    groundTruthTestSet1Np.append(torchToNp(groundTruthTestSet2[idx]))


showGridNumpyImg(inputsTestSet1Np, outMode1Np, groundTruthTestSet1Np, plotTitle='Modelo2+DATASET1+Norm')


######################## Testeo data set 2 ###############################

print('Data set 2')

getTestOneModelOneSlices(modelDT2,inputsTestSet2[30], groundTruthTestSet2[30], mse = 'True')
getTestOneModelOneSlices(modelDT2,inputsTestSet2[30], groundTruthTestSet2[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT2,inputsTestSet2[30], groundTruthTestSet2[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes
cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []

for idx in ramdomIdx :
    inputsTestSet2Np.append(torchToNp(inputsTestSet2[idx]))
    outModel = testModelSlice(modelDT2,inputsTestSet2[idx])
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0,:,:,:])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet2[idx]))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(modelDT2, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                       mseGrey='True', greyMatterValue=8)
    out, antesWhite, dpsWhite = getTestOneModelOneSlices(modelDT2, inputsTestSet2[idx], groundTruthTestSet2[idx],
                                                         mseWhite='True', whiteMatterValue=2)

    mseAntesGrey.append(antesGrey)
    mseDspGrey.append(dpsGrey)

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)



showGridNumpyImg(inputsTestSet2Np,outMode2Np,groundTruthTestSet2Np, plotTitle = 'Modelo2+DATASET2')

## EXCEL
df = pd.DataFrame()

modelo = np.ones([cantImg * 1])

df['Modelo'] = modelo
df['Slice Nro'] = ramdomIdx
df['MSE Grey antes'] = mseAntesGrey
df['MSE Grey dsp'] = mseDspGrey

df['MSE White antes'] = mseAntesWhite
df['MSE White dsp'] = mseDspWhite

df.to_excel('MODELO2+DATASET2.xlsx')

print('Data set 2 NORMALIZADO')

meanGroundTruth = np.mean(torchToNp(groundTruthTestSet2[30]))
meanInputs = np.mean(torchToNp(inputsTestSet2[30]))

newNormImage = (inputsTestSet2[30]/meanInputs)*meanGroundTruth

getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet2[30], mse = 'True')
getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet2[30], mseGrey = 'True', greyMatterValue= 8)
getTestOneModelOneSlices(modelDT2,newNormImage, groundTruthTestSet2[30], mseWhite = 'True', whiteMatterValue= 2)

## Imagenes NORMALIZADAS
cantImg = 3
cantIdx = (inputsTestSet2.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsTestSet2Np = []
outMode2Np = []
groundTruthTestSet2Np = []

for idx in ramdomIdx :
    meanGroundTruth = np.mean(torchToNp(groundTruthTestSet2[idx]))
    meanInputs = np.mean(torchToNp(inputsTestSet2[idx]))
    newNormImage = (inputsTestSet2[idx] / meanInputs) * meanGroundTruth

    inputsTestSet2Np.append(torchToNp(newNormImage))
    outModel = testModelSlice(modelDT2,newNormImage)
    outModel = torchToNp(outModel)
    outMode2Np.append(outModel[0,:,:,:])
    groundTruthTestSet2Np.append(torchToNp(groundTruthTestSet2[idx]))

showGridNumpyImg(inputsTestSet2Np,outMode2Np,groundTruthTestSet2Np, plotTitle = 'Modelo2+DATASET2+Norm')
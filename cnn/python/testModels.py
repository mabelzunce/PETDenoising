# Cargamos los modelos y visualizamos resultados

import torch
import torchvision
from unetM import Unet

from utils import reshapeDataSet
from utils import MSE
from utils import torchToImg
from utils import mseModelSlice
from utils import testModelSlice
from utils import obtenerMask

import SimpleITK as sitk
import numpy as np
from sklearn.model_selection import train_test_split

modelDT1 = Unet()
modelDT1.load_state_dict(torch.load('bestModelDataSet1_35'))

modelDT2 = Unet()
modelDT2.load_state_dict(torch.load('bestModelDataSet2_7'))

noisyDataSet1_nii = sitk.ReadImage('./noisyDataSet1.nii')
img_noisyDataSet1 = sitk.GetArrayFromImage(noisyDataSet1_nii)

noisyDataSet2_nii = sitk.ReadImage('./noisyDataSet2.nii')
img_noisyDataSet2 = sitk.GetArrayFromImage(noisyDataSet2_nii)

groundTruth_nii = sitk.ReadImage('./groundTruth.nii')
img_groundTruth = sitk.GetArrayFromImage(groundTruth_nii)

img_noisyDataSet1 = reshapeDataSet(img_noisyDataSet1)
img_noisyDataSet2 = reshapeDataSet(img_noisyDataSet2)
img_groundTruth = reshapeDataSet(img_groundTruth)

################################### MODELO 1 ##############################################
######################## Testeo data set 1 rotado ###############################

print('-------------------------------------------------')
print('------------MODEL 1------------------------------')
print('-------------------------------------------------')

trainingSet1 = dict([('input',img_noisyDataSet1), ('output', img_groundTruth)])

inputsTestSet1 = torch.from_numpy(trainingSet1['input'][:,:,:,:])
groundTruthTestSet1 = torch.from_numpy(trainingSet1['output'][:,:,:,:])

inputsTestSet1 = torchvision.transforms.functional.rotate(inputsTestSet1,15)
groundTruthTestSet1 = torchvision.transforms.functional.rotate(groundTruthTestSet1,15)

out = testModelSlice(modelDT1, inputsTestSet1[9])
out = (out).detach().numpy()
out = sitk.GetImageFromArray(out[0,:,:])
sitk.WriteImage(out,'out.nii')

mseBef, mseAft = mseModelSlice(modelDT1,inputsTestSet1[9],groundTruthTestSet1[9])

print('DATA SET 1 ROTADO')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

######################## Testeo data set 2 ###############################

trainingSet2 = dict([('input',img_noisyDataSet2), ('output', img_groundTruth)])

inputsTestSet2 = torch.from_numpy(trainingSet2['input'][:,:,:,:])
groundTruthTestSet2 = torch.from_numpy(trainingSet2['output'][:,:,:,:])

mseBef, mseAft = mseModelSlice(modelDT1,inputsTestSet2[30],groundTruthTestSet2[30])

print('DATA SET 2')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

######################## Testeo MATERIA GRIS y BLANCA DT1 ##############################

greyMatterValue = 8
whiteMatterValue = 2

nroSliceDS1 = 30

noisyOutDataSet1 = testModelSlice(modelDT1,inputsTestSet1[nroSliceDS1])

greyMaskMatterDataSet1 = obtenerMask(groundTruthTestSet1[nroSliceDS1],greyMatterValue)
whiteMaskMatterDataSet1 = obtenerMask(groundTruthTestSet1[nroSliceDS1],whiteMatterValue)

greyMatterNoisyDataSet1Antes = inputsTestSet1[nroSliceDS1] * greyMaskMatterDataSet1
greyMatterNoisyDataSet1Dsp = noisyOutDataSet1 * greyMaskMatterDataSet1

cantPix = np.count_nonzero(greyMaskMatterDataSet1)

mseBef, mseAft = mseModelSlice(modelDT1,greyMatterNoisyDataSet1Antes,greyMatterNoisyDataSet1Dsp,cantPix)

print('DATA SET 1 MATERIA GRIS')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

whiteMatterNoisyDataSet1Antes = inputsTestSet1[nroSliceDS1] * whiteMaskMatterDataSet1
whiteMatterNoisyDataSet1Dsp = noisyOutDataSet1 * whiteMaskMatterDataSet1

cantPix = np.count_nonzero(whiteMaskMatterDataSet1)

mseBef, mseAft = mseModelSlice(modelDT1,whiteMatterNoisyDataSet1Antes,whiteMatterNoisyDataSet1Dsp,cantPix)

print('DATA SET 1 MATERIA BLANCA')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)


noisyOutDataSet1= (noisyOutDataSet1).detach().numpy()
outModel1DataSet1 = sitk.GetImageFromArray(noisyOutDataSet1[0,:,:])
sitk.WriteImage(outModel1DataSet1,'outModel1DataSet1.nii')


maskWhiteModel1DataSet1 = sitk.GetImageFromArray(whiteMaskMatterDataSet1[0,:,:])
sitk.WriteImage(maskWhiteModel1DataSet1,'maskWhiteModel1DataSet1.nii')

######################## Testeo MATERIA GRIS y BLANCA DT2 ##############################

nroSliceDS2 = 30

noisyOutDataSet2 = testModelSlice(modelDT1,inputsTestSet2[nroSliceDS2])

greyMaskMatterDataSet2 = obtenerMask(groundTruthTestSet2[nroSliceDS2],greyMatterValue)
whiteMaskMatterDataSet2 = obtenerMask(groundTruthTestSet2[nroSliceDS2],whiteMatterValue)

greyMatterNoisyDataSet2Antes = inputsTestSet2[nroSliceDS2] * greyMaskMatterDataSet2
greyMatterNoisyDataSet2Dsp = noisyOutDataSet2 * greyMaskMatterDataSet2

cantPix = np.count_nonzero(greyMaskMatterDataSet2)

mseBef, mseAft = mseModelSlice(modelDT1,greyMatterNoisyDataSet2Antes,greyMatterNoisyDataSet2Dsp,cantPix)

print('DATA SET 2 MATERIA GRIS')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

whiteMatterNoisyDataSet2Antes = inputsTestSet2[nroSliceDS2] * whiteMaskMatterDataSet2
whiteMatterNoisyDataSet2Dsp = noisyOutDataSet2 * whiteMaskMatterDataSet2

cantPix = np.count_nonzero(whiteMaskMatterDataSet2)

mseBef, mseAft = mseModelSlice(modelDT1,whiteMatterNoisyDataSet2Antes,whiteMatterNoisyDataSet2Dsp,cantPix)


noisyOutDataSet2 = (noisyOutDataSet2).detach().numpy()
outModel1DataSet2 = sitk.GetImageFromArray(noisyOutDataSet2[0,:,:])
sitk.WriteImage(outModel1DataSet2,'outModel1DataSet2.nii')

maskWhiteModel1DataSet2 = sitk.GetImageFromArray(whiteMaskMatterDataSet2[0,:,:])
sitk.WriteImage(maskWhiteModel1DataSet2,'maskWhiteModel1DataSet2.nii')

print('DATA SET 2 MATERIA BLANCA')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)


################################### MODELO 2 ##############################################
######################## Testeo data set 2 rotado ###############################
print('-------------------------------------------------')
print('------------MODEL 2------------------------------')
print('-------------------------------------------------')

trainingSet1 = dict([('input',img_noisyDataSet2), ('output', img_groundTruth)])

inputsTestSet1 = torch.from_numpy(trainingSet1['input'][:,:,:,:])
groundTruthTestSet1 = torch.from_numpy(trainingSet1['output'][:,:,:,:])

inputsTestSet1 = torchvision.transforms.functional.rotate(inputsTestSet1,15)
groundTruthTestSet1 = torchvision.transforms.functional.rotate(groundTruthTestSet1,15)

out = testModelSlice(modelDT2, inputsTestSet1[9])
out = (out).detach().numpy()
out = sitk.GetImageFromArray(out[0,:,:])
sitk.WriteImage(out,'outModel2.nii')

mseBef, mseAft = mseModelSlice(modelDT2,inputsTestSet1[9],groundTruthTestSet1[9])

print('DATA SET 2 ROTADO')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

######################## Testeo data set 1 ###############################

trainingSet2 = dict([('input',img_noisyDataSet1), ('output', img_groundTruth)])

inputsTestSet2 = torch.from_numpy(trainingSet2['input'][:,:,:,:])
groundTruthTestSet2 = torch.from_numpy(trainingSet2['output'][:,:,:,:])

mseBef, mseAft = mseModelSlice(modelDT2,inputsTestSet2[30],groundTruthTestSet2[30])

print('DATA SET 1')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)


######################## Testeo MATERIA GRIS y BLANCA DT1 ##############################

nroSliceDS2 = 30

noisyOutDataSet2 = testModelSlice(modelDT2,inputsTestSet2[nroSliceDS2])

greyMaskMatterDataSet2 = obtenerMask(groundTruthTestSet2[nroSliceDS2],greyMatterValue)
whiteMaskMatterDataSet2 = obtenerMask(groundTruthTestSet2[nroSliceDS2],whiteMatterValue)

greyMatterNoisyDataSet2Antes = inputsTestSet2[nroSliceDS2] * greyMaskMatterDataSet2
greyMatterNoisyDataSet2Dsp = noisyOutDataSet2 * greyMaskMatterDataSet2

cantPix = np.count_nonzero(greyMaskMatterDataSet2)

mseBef, mseAft = mseModelSlice(modelDT2,greyMatterNoisyDataSet2Antes,greyMatterNoisyDataSet2Dsp,cantPix)

print('DATA SET 1 MATERIA GRIS')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

whiteMatterNoisyDataSet2Antes = inputsTestSet2[nroSliceDS2] * whiteMaskMatterDataSet2
whiteMatterNoisyDataSet2Dsp = noisyOutDataSet2 * whiteMaskMatterDataSet2

cantPix = np.count_nonzero(whiteMaskMatterDataSet2)

mseBef, mseAft = mseModelSlice(modelDT2,whiteMatterNoisyDataSet2Antes,whiteMatterNoisyDataSet2Dsp,cantPix)

noisyOutDataSet2 = (noisyOutDataSet2).detach().numpy()
outModel1DataSet2 = sitk.GetImageFromArray(noisyOutDataSet2[0,:,:])
sitk.WriteImage(outModel1DataSet2,'outModel2DataSet1.nii')

maskWhiteModel1DataSet2 = sitk.GetImageFromArray(whiteMaskMatterDataSet2[0,:,:])
sitk.WriteImage(maskWhiteModel1DataSet2,'maskWhiteModel2DataSet1.nii')

print('DATA SET 1 MATERIA BLANCA')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

######################## Testeo MATERIA GRIS y BLANCA DT2 ##############################

greyMatterValue = 8
whiteMatterValue = 2

nroSliceDS1 = 30

noisyOutDataSet1 = testModelSlice(modelDT2,inputsTestSet1[nroSliceDS1])

greyMaskMatterDataSet1 = obtenerMask(groundTruthTestSet1[nroSliceDS1],greyMatterValue)
whiteMaskMatterDataSet1 = obtenerMask(groundTruthTestSet1[nroSliceDS1],whiteMatterValue)

greyMatterNoisyDataSet1Antes = inputsTestSet1[nroSliceDS1] * greyMaskMatterDataSet1
greyMatterNoisyDataSet1Dsp = noisyOutDataSet1 * greyMaskMatterDataSet1

cantPix = np.count_nonzero(greyMaskMatterDataSet1)

mseBef, mseAft = mseModelSlice(modelDT2,greyMatterNoisyDataSet1Antes,greyMatterNoisyDataSet1Dsp,cantPix)

print('DATA SET 2 MATERIA GRIS')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)

whiteMatterNoisyDataSet1Antes = inputsTestSet1[nroSliceDS1] * whiteMaskMatterDataSet1
whiteMatterNoisyDataSet1Dsp = noisyOutDataSet1 * whiteMaskMatterDataSet1

cantPix = np.count_nonzero(whiteMaskMatterDataSet1)

mseBef, mseAft = mseModelSlice(modelDT2,whiteMatterNoisyDataSet1Antes,whiteMatterNoisyDataSet1Dsp,cantPix)

print('DATA SET 2 MATERIA BLANCA')
print('MSE antes de pasar por la red', mseBef)
print('MSE dsp de pasar por la red', mseAft)


noisyOutDataSet1= (noisyOutDataSet1).detach().numpy()
outModel1DataSet1 = sitk.GetImageFromArray(noisyOutDataSet1[0,:,:])
sitk.WriteImage(outModel1DataSet1,'outModel2DataSet2.nii')


maskWhiteModel1DataSet1 = sitk.GetImageFromArray(whiteMaskMatterDataSet1[0,:,:])
sitk.WriteImage(maskWhiteModel1DataSet1,'maskWhiteModel2DataSet2.nii')
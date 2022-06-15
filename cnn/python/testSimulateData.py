import torch
import torchvision
import skimage
import matplotlib as plt

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
from utils import covValue
from utils import crcValue

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

modelDT3 = Unet()
modelDT3.load_state_dict(torch.load('bestModelDataSet3_6'))


path = os.getcwd()
pathGroundTruth = path+'/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)

pathNoisyDataSet = path+'/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)

pathGreyMask = path+'/PhantomsGreyMask'
arrayGreyMask= os.listdir(pathGreyMask)

pathWhiteMask = path+'/PhantomsWhiteMask'
arrayWhiteMask= os.listdir(pathWhiteMask)

#calculo metricas por sujeto...
# leo el fantoma, el ground truth y las mascaras por sujeto

nameGroundTruth=[]
groundTruthArray = []

for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    groundTruth = sitk.ReadImage(pathGroundTruthElement)
    groundTruth = sitk.GetArrayFromImage(groundTruth)
    groundTruth = reshapeDataSet(groundTruth)
    groundTruthArray.append(groundTruth)
    nameGroundTruth.append(element[23:-4])

nameNoisyDataSet=[]
noisyImagesArray = []
for element in arrayNoisyDataSet:
    pathNoisyDataSetElement = pathNoisyDataSet+'/'+element
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    noisyDataSet = reshapeDataSet(noisyDataSet)
    noisyImagesArray.append(noisyDataSet)
    nameNoisyDataSet.append(element[21:-4])

namegreyMask=[]
greyMaskArray = []
for element in arrayGreyMask:
    pathGreyMaskElement = pathGreyMask+'/'+element
    greyMask = sitk.ReadImage(pathGreyMaskElement)
    greyMask = sitk.GetArrayFromImage(greyMask)
    greyMaskArray.append(greyMask)
    namegreyMask.append(element[7:-14])

nameWhiteMask=[]
whiteMaskArray = []
for element in arrayWhiteMask:
    pathWhiteMaskElement = pathWhiteMask+'/'+element
    whiteMask = sitk.ReadImage(pathWhiteMaskElement)
    whiteMask = sitk.GetArrayFromImage(whiteMask)
    whiteMaskArray.append(whiteMask)
    nameWhiteMask.append(element[7:-15])

# paso al modelo y evaluo resultados

noisyImagesArray = np.array(noisyImagesArray)
groundTruthArray = np.array(groundTruthArray)
whiteMaskArray = np.array(whiteMaskArray)
greyMaskArray = np.array(greyMaskArray)

noisyImagesArray = torch.from_numpy(noisyImagesArray)
groundTruthArray = torch.from_numpy(groundTruthArray)
whiteMaskArray = torch.from_numpy(whiteMaskArray)
greyMaskArray = torch.from_numpy(greyMaskArray)

covAntesSlice = []
covDspSlice = []

covAntesTodos = []
covDspTodos = []

crcAntesSlice = []
crcDspSlice = []

crcAntesTodos = []
crcDspTodos = []

inputsNp = []
outNp = []
groundTruthNp = []

meanValueAntes = []
meanValueDsp = []

outFilterSigma4 = []
outFilterSigma6 = []
crcSliceDspSigma4 = []
covSliceDspSigma4 = []
crcSliceDspSigma6 = []
covSliceDspSigma6 = []

covTodosDspSigma6 = []
covTodosDspSigma4 = []

crcTodosDspSigma6 = []
crcTodosDspSigma4 = []

subjectName = []
subjectSlice = []
subjectTotal = []

meanValueAntes = []
meanValueDsp = []
meanValueSigma4 = []
meanValueSigma6 = []

meanValueAntesSlice = []
meanValueDspSlice = []
meanValueSigma4Slice = []
meanValueSigma6Slice = []

mseValueAntes = []
mseValueDsp = []
mseValueSigma4 = []
mseValueSigma6 = []

mseGreyMatterAntes = []
mseGreyMatterDsp = []
mseGreyMatterSigma4 = []
mseGreyMatterSigma6 = []

mseValueSliceAntes = []
mseValueSliceDsp = []
mseValueSliceSigma4 = []
mseValueSliceSigma6 = []

for sub in range(0,len(noisyImagesArray)):

    inputsNp = []
    outNp = []
    groundTruthNp = []

    outFilterSigma4 = []
    outFilterSigma6 = []

    subjectNumbers= nameNoisyDataSet[sub]
    idxGreyMask = np.where((np.array(namegreyMask)) == subjectNumbers)
    greyMaskSubject = greyMaskArray[idxGreyMask]
    idxWhiteMask = np.where((np.array(nameWhiteMask)) == subjectNumbers)
    whiteMaskSubject = whiteMaskArray[idxWhiteMask]

    whiteMaskSubject = whiteMaskSubject[0,:,:,:]
    greyMaskSubject = greyMaskSubject[0,:,:,:]

    groundTruthSubject = groundTruthArray[sub,:,:,:]

    subjectImages = noisyImagesArray[sub,:,:,:]
    subjectName.append(subjectNumbers)

    print('Subject:',sub)

    for idx in range(0,len(subjectImages)):
        subjectSlice.append(idx)

        meanSliceInp = subjectImages[idx, :, :].mean()
        stdSliceInp = subjectImages[idx, :, :].std()

        if stdSliceInp!=0.0:
            inputsNorm = (subjectImages[idx] - meanSliceInp) / stdSliceInp
        else:
            inputsNorm = (subjectImages[idx] - meanSliceInp)

        outModel = testModelSlice(modelDT3, inputsNorm)

        if stdSliceInp != 0.0:
            outModelNp = torchToNp((outModel * stdSliceInp) + meanSliceInp)
        else:
            outModelNp = torchToNp((outModel) + meanSliceInp)

        inputsNp.append(torchToNp(subjectImages[idx]))
        groundTruthNp.append(torchToNp(groundTruthSubject[idx]))
        outNp.append(outModelNp[0, :, :, :])

        outFilterSigma4.append(skimage.filters.gaussian(subjectImages[idx, :, :], sigma=(4 / 2.35)))
        crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilterSigma4[-1][0,:,:]), greyMaskSubject[idx], whiteMaskSubject[idx]))
        covSliceDspSigma4.append(covValue(torch.from_numpy(outFilterSigma4[-1][0,:,:]), greyMaskSubject[idx]))

        outFilterSigma6.append(skimage.filters.gaussian(subjectImages[idx, :, :], sigma=(6/2.35)))
        crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilterSigma6[-1][0,:,:]), greyMaskSubject[idx], whiteMaskSubject[idx]))
        covSliceDspSigma6.append(covValue(torch.from_numpy(outFilterSigma6[-1][0,:,:]), greyMaskSubject[idx]))

        covAntesSlice.append(covValue(subjectImages[idx,0,:,:], greyMaskSubject[idx]))
        crcAntesSlice.append(crcValue(subjectImages[idx,0,:,:], greyMaskSubject[idx], whiteMaskSubject[idx]))

        covDspSlice.append(covValue(torch.from_numpy(np.array(outNp)[idx, 0, 0,:, :]), greyMaskSubject[idx]))
        crcDspSlice.append(crcValue(torch.from_numpy(np.array(outNp)[idx, 0, 0,:, :]), greyMaskSubject[idx],whiteMaskSubject[idx]))

        meanValueAntesSlice.append(np.mean(np.trim_zeros(((torch.Tensor.numpy(subjectImages)[idx,0,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx])).flatten())))
        meanValueDspSlice.append(np.mean(np.trim_zeros(((np.array(outNp)[idx,0,0,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx])).flatten())))
        meanValueSigma4Slice.append(np.mean(np.trim_zeros(((np.array(outFilterSigma4)[idx,0,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx])).flatten())))
        meanValueSigma6Slice.append(np.mean(np.trim_zeros(((np.array(outFilterSigma6)[idx, 0, :, :]) * torch.Tensor.numpy(greyMaskSubject[idx])).flatten())))

        mseValueSliceAntes.append(MSE(torch.Tensor.numpy(subjectImages[idx, 0, :, :]), torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]),cantPixels = 256*256))
        mseValueSliceDsp.append(MSE((np.array(outNp))[idx, 0, 0, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))
        mseValueSliceSigma4.append(MSE((np.array(outFilterSigma4))[idx, 0, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))
        mseValueSliceSigma6.append(MSE((np.array(outFilterSigma6))[idx, 0, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))

        subjectTotal.append(subjectNumbers)


    meanValueAntes.append(np.mean(np.trim_zeros(((torch.Tensor.numpy(subjectImages)[:,0,:,:]) * torch.Tensor.numpy(greyMaskSubject)).flatten())))
    meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outNp)[:,0,0,:,:]) * torch.Tensor.numpy(greyMaskSubject)).flatten())))
    meanValueSigma4.append(np.mean(np.trim_zeros(((np.array(outFilterSigma4)[:,0,:,:]) * torch.Tensor.numpy(greyMaskSubject)).flatten())))
    meanValueSigma6.append(np.mean(np.trim_zeros(((np.array(outFilterSigma6)[:, 0, :, :]) * torch.Tensor.numpy(greyMaskSubject)).flatten())))

    mseValueAntes.append(MSE(torch.Tensor.numpy(subjectImages[:,0,:,:]),torch.Tensor.numpy(groundTruthSubject[:,0,:,:])))
    mseValueDsp.append(MSE((np.array(outNp))[:, 0, 0, :, :], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))
    mseValueSigma4.append(MSE((np.array(outFilterSigma4))[:,0,:,:], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))
    mseValueSigma6.append(MSE((np.array(outFilterSigma6))[:, 0, :, :], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))

    mseGreyMatterAntes.append(MSE(torch.Tensor.numpy(subjectImages[:, 0, :, :]), torch.Tensor.numpy(greyMaskSubject)))
    mseGreyMatterDsp.append(MSE((np.array(outNp))[:, 0, 0, :, :], torch.Tensor.numpy(greyMaskSubject)))
    mseGreyMatterSigma4.append(MSE((np.array(outFilterSigma4))[:, 0, :, :], torch.Tensor.numpy(greyMaskSubject)))
    mseGreyMatterSigma6.append(MSE((np.array(outFilterSigma6))[:, 0, :, :], torch.Tensor.numpy(greyMaskSubject)))

    covAntesTodos.append(covValue((torch.from_numpy(np.array(inputsNp)))[:, 0,0, :, :], greyMaskSubject))
    crcAntesTodos.append(crcValue((torch.from_numpy(np.array(inputsNp)))[:, 0,0, :, :], greyMaskSubject, whiteMaskSubject))

    covDspTodos.append(covValue((torch.from_numpy(np.array(outNp)))[:, 0,0, :, :], greyMaskSubject))
    crcDspTodos.append(crcValue((torch.from_numpy(np.array(outNp)))[:, 0,0, :, :], greyMaskSubject, whiteMaskSubject))

    covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilterSigma6)[:,0,:,:]), greyMaskSubject))
    covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilterSigma4)[:,0,:,:]), greyMaskSubject))

    crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilterSigma6)[:,0,:,:]), greyMaskSubject, whiteMaskSubject))
    crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilterSigma4)[:,0,:,:]), greyMaskSubject, whiteMaskSubject))


    #image = sitk.GetImageFromArray(np.array(inputsNp)[:, 0,0, :, :])
    #nameImage = 'Subject' + subjectNumbers + 'Input1%.nii'
    #sitk.WriteImage(image, nameImage)

    #image = sitk.GetImageFromArray(np.array(outNp)[:, 0,0, :, :])
    #nameImage = 'Subject' + subjectNumbers + 'Out1%.nii'
    #sitk.WriteImage(image, nameImage)

    #image = sitk.GetImageFromArray(np.array(outFilterSigma4)[:, 0, :, :])
    #nameImage = 'Subject' + subjectNumbers + 'FilterSigma4-1%.nii'
    #sitk.WriteImage(image, nameImage)

    #image = sitk.GetImageFromArray(np.array(outFilterSigma6)[:, 0, :, :])
    #nameImage = 'Subject' + subjectNumbers + 'FilterSigma6-1%.nii'
    #sitk.WriteImage(image, nameImage)



## Guardar en Excel
dfGlobal = pd.DataFrame()

dfGlobal['COV antes'] = covAntesTodos
dfGlobal['COV dsp'] = covDspTodos
dfGlobal['COV dsp Filtro Sigma 4'] = covTodosDspSigma4
dfGlobal['COV dsp Filtro Sigma 6'] = covTodosDspSigma6

dfGlobal['CRC antes'] = crcAntesTodos
dfGlobal['CRC dsp'] = crcDspTodos
dfGlobal['CRC dsp Filtro Sigma 4'] = crcTodosDspSigma4
dfGlobal['CRC dsp Filtro Sigma 6'] = crcTodosDspSigma6

dfGlobal['Mean Grey Matter antes'] = meanValueAntes
dfGlobal['Mean Grey Matter dsp model'] = meanValueDsp
dfGlobal['Mean Grey Matter dsp sigma4'] = meanValueSigma4
dfGlobal['Mean Grey Matter dsp sigma6'] = meanValueSigma6

dfGlobal['MSE  antes'] = mseValueAntes
dfGlobal['MSE dsp model'] = mseValueDsp
dfGlobal['MSE dsp sigma4'] = mseValueSigma4
dfGlobal['MSE dsp sigma6'] = mseValueSigma6

dfGlobal['MSE  GM antes'] = mseGreyMatterAntes
dfGlobal['MSE GM dsp model'] = mseGreyMatterDsp
dfGlobal['MSE GM dsp sigma4'] = mseGreyMatterSigma4
dfGlobal['MSE GM dsp sigma6'] = mseGreyMatterSigma6

dfGlobal['Subject'] = subjectName

dfGlobal.to_excel('SimulateDataWithModel3Subject5%New.xlsx')

## Guardar en Excel
dfSlice = pd.DataFrame()

dfSlice['Slice'] = subjectSlice
dfSlice['Subject'] = subjectTotal

dfSlice['COV antes'] = covAntesSlice
dfSlice['COV dsp model'] = covDspSlice
dfSlice['COV dsp sigma4'] = covSliceDspSigma4
dfSlice['COV dsp sigma6'] = covSliceDspSigma6

dfSlice['CRC antes'] = crcAntesSlice
dfSlice['CRC dsp'] = crcDspSlice
dfSlice['CRC dsp sigma4'] = crcSliceDspSigma4
dfSlice['CRC dsp sigma6'] = crcSliceDspSigma6

dfSlice['MSE  antes'] = mseValueSliceAntes
dfSlice['MSE dsp model'] = mseValueSliceDsp
dfSlice['MSE dsp sigma4'] = mseValueSliceSigma4
dfSlice['MSE dsp sigma6'] = mseValueSliceSigma6

dfSlice['Mean Grey Matter antes'] = meanValueAntesSlice
dfSlice['Mean Grey Matter dsp model'] = meanValueDspSlice
dfSlice['Mean Grey Matter dsp sigma4'] = meanValueSigma4Slice
dfSice['Mean Grey Matter dsp sigma6'] = meanValueSigma6Slice

# dfSlice['Subject'] = pet

dfSlice.to_excel('SimulateDataWithModel3SubjectSlices5%New.xlsx')
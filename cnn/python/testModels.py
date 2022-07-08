import torch
import torchvision
import skimage
import matplotlib.pyplot as plt

#from unetM import Unet
from unet import Unet
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

# Testeo todos lo modelos...

model = Unet(1,1)

path = os.getcwd()

#modelsPath = path+'/Modelo7'
modelsPath = path+'/ModeloUnetResidualUno'
#modelsPath = path+'/Modelo8'
models = os.listdir(modelsPath)

pathSaveResults = "C:/Users/Encargado/Desktop/RESULTADOS/UnetResidualUno"

pathGroundTruth = path+'/newDataset/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)

pathNoisyDataSet = path+'/newDataset/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)

pathGreyMask = path+'/newDataset/maskGreyMatter'
arrayGreyMask= os.listdir(pathGreyMask)

pathWhiteMask = path+'/newDataset/maskWhiteMatter'
arrayWhiteMask= os.listdir(pathWhiteMask)

nameGroundTruth=[]
groundTruthArray = []
noisyImagesArray = []
greyMaskArray = []
whiteMaskArray = []

validSubjects = [10,19]
#validSubjects = [8,17]
#validSubjects = [1,5]
trainSubjects = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,20]

conjunto = 'Train'
excluirConjunto = validSubjects

nameModel = 'UnetResidualUno'

nameConjunto = []

# leo los dataSet
for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element

    # read GroundTruth
    groundTruthImg = sitk.ReadImage(pathGroundTruthElement)
    groundTruth = sitk.GetArrayFromImage(groundTruthImg)
    groundTruth = reshapeDataSet(groundTruth)

    name, extension = os.path.splitext(element)
    if extension == '.nii':
        name, extension2 = os.path.splitext(name)
    ind = name.find('Subject')
    name = name[ind + len('Subject'):]

    nameGroundTruth.append(name)

    # read noisyDataSet
    nametrainNoisyDataSet = 'noisyDataSet5_Subject' + name + '.nii'
    pathNoisyDataSetElement = pathNoisyDataSet + '/' + nametrainNoisyDataSet
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    noisyDataSet = reshapeDataSet(noisyDataSet)

    # read greyMask
    nameGreyMask = 'Subject' + name + 'GreyMask.nii'
    pathGreyMaskElement = pathGreyMask + '/' + nameGreyMask
    greyMask = sitk.ReadImage(pathGreyMaskElement)
    greyMask = sitk.GetArrayFromImage(greyMask)
    greyMask = reshapeDataSet(greyMask)

    # read whiteMask
    nameWhiteMask = 'Subject' + name + 'WhiteMask.nii'
    pathWhiteMaskElement = pathWhiteMask + '/' + nameWhiteMask
    whiteMask = sitk.ReadImage(pathWhiteMaskElement)
    whiteMask = sitk.GetArrayFromImage(whiteMask)
    whiteMask = reshapeDataSet(whiteMask)

    if int(name) not in excluirConjunto:
        groundTruthArray.append(groundTruth)
        noisyImagesArray.append(noisyDataSet)
        greyMaskArray.append(greyMask)
        whiteMaskArray.append(whiteMask)
        nameConjunto.append(name)

# paso al modelo y evaluo resultados
noisyImagesArray = np.array(noisyImagesArray)
groundTruthArray = np.array(groundTruthArray)
whiteMaskArray = np.array(whiteMaskArray)
greyMaskArray = np.array(greyMaskArray)

noisyImagesArray = torch.from_numpy(noisyImagesArray)
groundTruthArray = torch.from_numpy(groundTruthArray)
whiteMaskArray = torch.from_numpy(whiteMaskArray)
greyMaskArray = torch.from_numpy(greyMaskArray)

# evaluo todos los modelos
allModelsCovAntes = []
allModelsCovDsp = []
allModelsCovDspSigma4 = []
allModelsCovDspSigma6 = []
allModelsCovDspSigma8 = []

allModelsCrcAntesTodos = []
allModelsCrcDsp = []
allModelsCrcDspSigma4 = []
allModelsCrcDspSigma6 = []
allModelsCrcDspSigma8 = []

allModelsMeanValueAntes = []
allModelsMeanValueDsp = []
allModelsMeanValueSigma4 = []
allModelsMeanValueSigma6 = []
allModelsMeanValueSigma8 = []

allModelsMeanValueAntesWhiteMatter = []
allModelsMeanValueDspWhiteMatter = []
allModelsMeanValueSigma4WhiteMatter = []
allModelsMeanValueSigma6WhiteMatter = []
allModelsMeanValueSigma8WhiteMatter = []

allModelsMseValueAntes = []
allModelsMseValueDsp = []
allModelsMseValueSigma4 = []
allModelsMseValueSigma6 = []
allModelsMseValueSigma8 = []

allModelsMseGreyMatterAntes = []
allModelsMseGreyMatterDsp = []
allModelsMseGreyMatterSigma4 = []
allModelsMseGreyMatterSigma6 = []
allModelsMseGreyMatterSigma8 = []

allModelsMseWhiteMatterAntes = []
allModelsMseWhiteMatterDsp = []
allModelsMseWhiteMatterSigma4 = []
allModelsMseWhiteMatterSigma6 = []
allModelsMseWhiteMatterSigma8 = []

allModelsStdValueAntes = []
allModelsStdValueDsp = []
allModelsStdValueSigma4 = []
allModelsStdValueSigma6 = []
allModelsStdValueSigma8 = []

modelName = []
contModel = 0

for nroModel in models:
    contModel = contModel+ 1
    model.load_state_dict(torch.load(modelsPath+'/'+nroModel, map_location=torch.device('cpu')))
    modelName.append(nroModel)

    crcSliceDspSigma4 = []
    covSliceDspSigma4 = []

    outFilterSigma6 = []
    crcSliceDspSigma6 = []
    covSliceDspSigma6 = []

    outFilterSigma8 = []
    crcSliceDspSigma8 = []
    covSliceDspSigma8 = []

    covAntesSlice = []
    crcAntesSlice = []

    covDspSlice = []
    crcDspSlice = []

    meanValueAntesSlice = []
    meanValueDspSlice = []
    meanValueSigma4Slice = []
    meanValueSigma6Slice = []
    meanValueSigma8Slice = []

    meanValueAntesWhiteMatter = []
    meanValueDspWhiteMatter = []
    meanValueSigma4WhiteMatter = []
    meanValueSigma6WhiteMatter = []
    meanValueSigma8WhiteMatter = []

    mseValueSliceAntes = []
    mseValueSliceDsp = []
    mseValueSliceSigma4 = []
    mseValueSliceSigma6 = []
    mseValueSliceSigma8 = []

    meanValueAntes = []
    stdValueAntes = []

    meanValueDsp = []
    stdValueDsp = []

    meanValueSigma4 = []
    stdValueSigma4 = []

    meanValueSigma6 = []
    stdValueSigma6 = []

    meanValueSigma8 = []
    stdValueSigma8 = []

    mseValueAntes = []
    mseValueDsp = []
    mseValueSigma4 = []
    mseValueSigma6 = []
    mseValueSigma8 = []

    mseGreyMatterAntes = []
    mseGreyMatterDsp = []
    mseGreyMatterSigma4 = []
    mseGreyMatterSigma6 = []
    mseGreyMatterSigma8 = []

    mseWhiteMatterAntes = []
    mseWhiteMatterDsp = []
    mseWhiteMatterSigma4 = []
    mseWhiteMatterSigma6 = []
    mseWhiteMatterSigma8 = []

    covAntesTodos = []
    crcAntesTodos = []

    covDspTodos = []
    crcDspTodos = []

    covTodosDspSigma6 = []
    covTodosDspSigma4 = []
    covTodosDspSigma8 = []

    crcTodosDspSigma6 = []
    crcTodosDspSigma4 = []
    crcTodosDspSigma8 = []

    subjectTotal = []
    subjectName = []
    subjectSlice = []

    print('Modelo {} de {}'.format(contModel, len(models)))

    for sub in range(0,len(noisyImagesArray)):

        inputsNp = []
        outNp = []
        groundTruthNp = []

        outFilterSigma4 = []
        outFilterSigma6 = []
        outFilterSigma8 = []

        subjectNumbers= nameConjunto[sub]
        subjectName.append(subjectNumbers)

        subjectImages = noisyImagesArray[sub, :, :, :]
        whiteMaskSubject = whiteMaskArray[sub, :, :, :]
        greyMaskSubject = greyMaskArray[sub, :, :, :]
        groundTruthSubject = groundTruthArray[sub,:,:,:]

        print('Subject:',nameConjunto[sub])

        for idx in range(0,len(subjectImages)):
            subjectSlice.append(idx)

            maxSlice = subjectImages[idx,0 ,:, :].max()

            if (maxSlice > 0.0):
                inputsNorm = ((subjectImages[idx]) / maxSlice)
            else:
                inputsNorm = ((subjectImages[idx]))

            outModel = testModelSlice(model, inputsNorm)

            if (maxSlice > 0.0):
                outModelNp = torchToNp((outModel * maxSlice))
            else:
                outModelNp = torchToNp((outModel *maxSlice))

            inputsNp.append(torchToNp(subjectImages[idx]))
            groundTruthNp.append(torchToNp(groundTruthSubject[idx]))
            outNp.append(outModelNp[0, :, :, :])

            outFilterSigma4.append(skimage.filters.gaussian(subjectImages[idx,0 ,:, :], sigma=(2 / 2.35)))
            crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilterSigma4[-1]), greyMaskSubject[idx,0,:,:], whiteMaskSubject[idx,0,:,:]))
            covSliceDspSigma4.append(covValue(torch.from_numpy(outFilterSigma4[-1]), greyMaskSubject[idx,0,:,:]))

            outFilterSigma6.append(skimage.filters.gaussian(subjectImages[idx, 0, :, :], sigma=(3 / 2.35)))
            crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilterSigma6[-1]), greyMaskSubject[idx, 0, :, :],
                                              whiteMaskSubject[idx, 0, :, :]))
            covSliceDspSigma6.append(covValue(torch.from_numpy(outFilterSigma6[-1]), greyMaskSubject[idx, 0, :, :]))

            outFilterSigma8.append(skimage.filters.gaussian(subjectImages[idx,0 ,:, :], sigma=(4/2.35)))
            crcSliceDspSigma8.append(crcValue(torch.from_numpy(outFilterSigma8[-1]), greyMaskSubject[idx,0,:,:], whiteMaskSubject[idx,0,:,:]))
            covSliceDspSigma8.append(covValue(torch.from_numpy(outFilterSigma8[-1]), greyMaskSubject[idx,0,:,:]))

            covAntesSlice.append(covValue(subjectImages[idx,0,:,:], greyMaskSubject[idx,0,:,:]))
            crcAntesSlice.append(crcValue(subjectImages[idx,0,:,:], greyMaskSubject[idx,0,:,:], whiteMaskSubject[idx,0,:,:]))

            covDspSlice.append(covValue(torch.from_numpy(np.array(outNp)[idx, 0, 0,:, :]), greyMaskSubject[idx,0,:,:]))
            crcDspSlice.append(crcValue(torch.from_numpy(np.array(outNp)[idx, 0, 0,:, :]), greyMaskSubject[idx,0,:,:],whiteMaskSubject[idx,0,:,:]))

            calcularMean = ((torch.Tensor.numpy(subjectImages)[idx,0,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx,0,:,:])).flatten()
            calcularMean = calcularMean[calcularMean != 0.0]
            meanValueAntesSlice.append(np.mean(calcularMean))
            calcularMean = ((np.array(outNp)[idx,0,0,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx,0,:,:])).flatten()
            calcularMean = calcularMean[calcularMean != 0.0]
            meanValueDspSlice.append(np.mean(calcularMean))
            calcularMean = (((np.array(outFilterSigma4)[idx,:,:]) * torch.Tensor.numpy(greyMaskSubject[idx,0,:,:])).flatten())
            calcularMean = calcularMean[calcularMean != 0.0]
            meanValueSigma4Slice.append(np.mean(calcularMean))
            calcularMean = ((((np.array(outFilterSigma6)[idx, :, :]) * torch.Tensor.numpy(greyMaskSubject[idx,0,:,:])).flatten()))
            calcularMean = calcularMean[calcularMean != 0.0]
            meanValueSigma6Slice.append(np.mean(calcularMean))
            calcularMean = ((((np.array(outFilterSigma8)[idx, :, :]) * torch.Tensor.numpy(greyMaskSubject[idx, 0, :, :])).flatten()))
            calcularMean = calcularMean[calcularMean != 0.0]
            meanValueSigma8Slice.append(np.mean(calcularMean))

            mseValueSliceAntes.append(MSE(torch.Tensor.numpy(subjectImages[idx, 0, :, :]), torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]),cantPixels = 256*256))
            mseValueSliceDsp.append(MSE((np.array(outNp))[idx, 0, 0, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))
            mseValueSliceSigma4.append(MSE((np.array(outFilterSigma4))[idx, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))
            mseValueSliceSigma6.append(MSE((np.array(outFilterSigma6))[idx, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]), cantPixels = 256*256))
            mseValueSliceSigma8.append(MSE((np.array(outFilterSigma8))[idx, :, :], torch.Tensor.numpy(groundTruthSubject[idx, 0, :, :]),cantPixels=256 * 256))

            subjectTotal.append(subjectNumbers)

        calcularMean = (((torch.Tensor.numpy(subjectImages)[:,0,:,:]) * torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueAntes.append(np.mean(calcularMean))
        stdValueAntes.append(np.std(calcularMean))

        calcularMean = (((np.array(outNp)[:,0,0,:,:]) * torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueDsp.append(np.mean(calcularMean))
        stdValueDsp.append(np.std(calcularMean))

        calcularMean = (((np.array(outFilterSigma4)[:,:,:]) * torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma4.append(np.mean(calcularMean))
        stdValueSigma4.append(np.std(calcularMean))

        calcularMean = (((np.array(outFilterSigma6)[:,  :, :]) * torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma6.append(np.mean(calcularMean))
        stdValueSigma6.append(np.std(calcularMean))

        calcularMean = (((np.array(outFilterSigma8)[:, :, :]) * torch.Tensor.numpy(greyMaskSubject)[:, 0, :, :]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma8.append(np.mean(calcularMean))
        stdValueSigma8.append(np.std(calcularMean))

        calcularMean = (((torch.Tensor.numpy(subjectImages)[:, 0, :, :]) * torch.Tensor.numpy(whiteMaskSubject)[:, 0, :,:]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueAntesWhiteMatter.append(np.mean(calcularMean))

        calcularMean = (((np.array(outNp)[:, 0, 0, :, :]) * torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueDspWhiteMatter.append(np.mean(calcularMean))

        calcularMean = (((np.array(outFilterSigma4)[:, :, :]) * torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma4WhiteMatter.append(np.mean(calcularMean))

        calcularMean = (((np.array(outFilterSigma6)[:, :, :]) * torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma6WhiteMatter.append(np.mean(calcularMean))

        calcularMean = (((np.array(outFilterSigma8)[:, :, :]) * torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]).flatten())
        calcularMean = calcularMean[calcularMean != 0.0]
        meanValueSigma8WhiteMatter.append(np.mean(calcularMean))

        mseValueAntes.append(MSE(torch.Tensor.numpy(subjectImages[:,0,:,:]),torch.Tensor.numpy(groundTruthSubject[:,0,:,:])))
        mseValueDsp.append(MSE((np.array(outNp))[:, 0, 0, :, :], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))
        mseValueSigma4.append(MSE((np.array(outFilterSigma4))[:,:,:], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))
        mseValueSigma6.append(MSE((np.array(outFilterSigma6))[:,:, :], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))
        mseValueSigma8.append(MSE((np.array(outFilterSigma8))[:, :, :], torch.Tensor.numpy(groundTruthSubject[:, 0, :, :])))

        mseGreyMatterAntes.append(MSE(torch.Tensor.numpy(subjectImages[:, 0, :, :]), torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]))
        mseGreyMatterDsp.append(MSE((np.array(outNp))[:, 0, 0, :, :], torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]))
        mseGreyMatterSigma4.append(MSE((np.array(outFilterSigma4))[:, :, :], torch.Tensor.numpy(greyMaskSubject)[:,0,:,:]))
        mseGreyMatterSigma6.append(MSE((np.array(outFilterSigma6))[:, :, :], torch.Tensor.numpy(greyMaskSubject[:,0,:,:])))
        mseGreyMatterSigma8.append(MSE((np.array(outFilterSigma8))[:, :, :], torch.Tensor.numpy(greyMaskSubject[:, 0, :, :])))

        mseWhiteMatterAntes.append(MSE(torch.Tensor.numpy(subjectImages[:, 0, :, :]), torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]))
        mseWhiteMatterDsp.append(MSE((np.array(outNp))[:, 0, 0, :, :], torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]))
        mseWhiteMatterSigma4.append(MSE((np.array(outFilterSigma4))[:, :, :], torch.Tensor.numpy(whiteMaskSubject)[:, 0, :, :]))
        mseWhiteMatterSigma6.append(MSE((np.array(outFilterSigma6))[:, :, :], torch.Tensor.numpy(whiteMaskSubject[:, 0, :, :])))
        mseWhiteMatterSigma8.append(MSE((np.array(outFilterSigma8))[:, :, :], torch.Tensor.numpy(whiteMaskSubject[:, 0, :, :])))

        covAntesTodos.append(covValue((torch.from_numpy(np.array(inputsNp)))[:, 0,0, :, :], greyMaskSubject[:,0,:,:]))
        crcAntesTodos.append(crcValue((torch.from_numpy(np.array(inputsNp)))[:, 0,0, :, :], greyMaskSubject[:,0,:,:], whiteMaskSubject[:,0,:,:]))

        covDspTodos.append(covValue((torch.from_numpy(np.array(outNp)))[:, 0,0, :, :], greyMaskSubject[:,0,:,:]))
        crcDspTodos.append(crcValue((torch.from_numpy(np.array(outNp)))[:, 0,0, :, :], greyMaskSubject[:,0,:,:], whiteMaskSubject[:,0,:,:]))

        covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilterSigma6)[:,:,:]), greyMaskSubject[:,0,:,:]))
        covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilterSigma4)[:,:,:]), greyMaskSubject[:,0,:,:]))
        covTodosDspSigma8.append(covValue(torch.from_numpy(np.array(outFilterSigma8)[:, :, :]), greyMaskSubject[:, 0, :, :]))

        crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilterSigma6)[:,:,:]), greyMaskSubject[:,0,:,:], whiteMaskSubject[:,0,:,:]))
        crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilterSigma4)[:,:,:]), greyMaskSubject[:,0,:,:], whiteMaskSubject[:,0,:,:]))
        crcTodosDspSigma8.append(crcValue(torch.from_numpy(np.array(outFilterSigma8)[:, :, :]), greyMaskSubject[:, 0, :, :],whiteMaskSubject[:, 0, :, :]))

        voxelSize = groundTruthImg.GetSpacing()

        image = sitk.GetImageFromArray(np.array(inputsNp)[:, 0, 0, :, :])
        image.SetSpacing(voxelSize)
        nameImage = 'Subject' + subjectNumbers + 'Input5%'+nroModel+'.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)

        image = sitk.GetImageFromArray(np.array(outNp)[:, 0,0, :, :])
        image.SetSpacing(voxelSize)
        nameImage = 'Subject' + subjectNumbers + 'Outpu5%'+nroModel+'.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)

        image = sitk.GetImageFromArray(np.array(outFilterSigma4)[:, :, :])
        image.SetSpacing(voxelSize)
        nameImage = 'Subject' + subjectNumbers + 'FilterSigma4-5%'+nroModel+'.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)

        image = sitk.GetImageFromArray(np.array(outFilterSigma6)[:, :, :])
        image.SetSpacing(voxelSize)
        nameImage = 'Subject' + subjectNumbers + 'FilterSigma6-5%'+nroModel+'.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)

        image = sitk.GetImageFromArray(np.array(outFilterSigma8)[:, :, :])
        image.SetSpacing(voxelSize)
        nameImage = 'Subject' + subjectNumbers + 'FilterSigma8-5%'+nroModel+'.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)

        ## Guardar en Excel
        dfGlobal = pd.DataFrame()

        dfGlobal['COV antes'] = covAntesTodos
        dfGlobal['COV dsp'] = covDspTodos
        dfGlobal['COV dsp Filtro Sigma 4'] = covTodosDspSigma4
        dfGlobal['COV dsp Filtro Sigma 6'] = covTodosDspSigma6
        dfGlobal['COV dsp Filtro Sigma 8'] = covTodosDspSigma8

        dfGlobal['CRC antes'] = crcAntesTodos
        dfGlobal['CRC dsp'] = crcDspTodos
        dfGlobal['CRC dsp Filtro Sigma 4'] = crcTodosDspSigma4
        dfGlobal['CRC dsp Filtro Sigma 6'] = crcTodosDspSigma6
        dfGlobal['CRC dsp Filtro Sigma 8'] = crcTodosDspSigma8

        dfGlobal['Mean GM antes'] = meanValueAntes
        dfGlobal['Mean GM dsp model'] = meanValueDsp
        dfGlobal['Mean GM dsp sigma4'] = meanValueSigma4
        dfGlobal['Mean GM dsp sigma6'] = meanValueSigma6
        dfGlobal['Mean GM dsp sigma8'] = meanValueSigma8

        dfGlobal['Mean WM antes'] = meanValueAntesWhiteMatter
        dfGlobal['Mean WM dsp model'] = meanValueDspWhiteMatter
        dfGlobal['Mean WM dsp sigma4'] = meanValueSigma4WhiteMatter
        dfGlobal['Mean WM dsp sigma6'] = meanValueSigma6WhiteMatter
        dfGlobal['Mean WM dsp sigma8'] = meanValueSigma8WhiteMatter

        dfGlobal['MSE antes'] = mseValueAntes
        dfGlobal['MSE dsp model'] = mseValueDsp
        dfGlobal['MSE dsp sigma4'] = mseValueSigma4
        dfGlobal['MSE dsp sigma6'] = mseValueSigma6
        dfGlobal['MSE dsp sigma8'] = mseValueSigma8

        dfGlobal['MSE GM antes'] = mseGreyMatterAntes
        dfGlobal['MSE GM dsp model'] = mseGreyMatterDsp
        dfGlobal['MSE GM dsp sigma4'] = mseGreyMatterSigma4
        dfGlobal['MSE GM dsp sigma6'] = mseGreyMatterSigma6
        dfGlobal['MSE GM dsp sigma8'] = mseGreyMatterSigma8

        dfGlobal['MSE WM antes'] = mseWhiteMatterAntes
        dfGlobal['MSE WM dsp model'] = mseWhiteMatterDsp
        dfGlobal['MSE WM dsp sigma4'] = mseWhiteMatterSigma4
        dfGlobal['MSE WM dsp sigma6'] = mseWhiteMatterSigma6
        dfGlobal['MSE WM dsp sigma8'] = mseWhiteMatterSigma8

        dfGlobal['std GM antes'] = stdValueAntes
        dfGlobal['std GM dsp model'] = stdValueDsp
        dfGlobal['std GM dsp sigma4'] = stdValueSigma4
        dfGlobal['std GM dsp sigma6'] = stdValueSigma6
        dfGlobal['std GM dsp sigma8'] = stdValueSigma8

        dfGlobal['Subject'] = subjectName

        name = 'SimulateDataWith'+nroModel+conjunto+'Subject5%.xlsx'

        dfGlobal.to_excel(pathSaveResults+'/'+name)

        ## Guardar en Excel
        dfSlice = pd.DataFrame()

        dfSlice['Slice'] = subjectSlice
        dfSlice['Subject'] = subjectTotal

        dfSlice['COV antes'] = covAntesSlice
        dfSlice['COV dsp model'] = covDspSlice
        dfSlice['COV dsp sigma4'] = covSliceDspSigma4
        dfSlice['COV dsp sigma6'] = covSliceDspSigma6
        dfSlice['COV dsp sigma8'] = covSliceDspSigma8

        dfSlice['CRC antes'] = crcAntesSlice
        dfSlice['CRC dsp'] = crcDspSlice
        dfSlice['CRC dsp sigma4'] = crcSliceDspSigma4
        dfSlice['CRC dsp sigma6'] = crcSliceDspSigma6
        dfSlice['CRC dsp sigma8'] = crcSliceDspSigma8

        dfSlice['MSE antes'] = mseValueSliceAntes
        dfSlice['MSE dsp model'] = mseValueSliceDsp
        dfSlice['MSE dsp sigma4'] = mseValueSliceSigma4
        dfSlice['MSE dsp sigma6'] = mseValueSliceSigma6
        dfSlice['MSE dsp sigma8'] = mseValueSliceSigma8

        dfSlice['Mean Grey Matter antes'] = meanValueAntesSlice
        dfSlice['Mean Grey Matter dsp model'] = meanValueDspSlice
        dfSlice['Mean Grey Matter dsp sigma4'] = meanValueSigma4Slice
        dfSlice['Mean Grey Matter dsp sigma6'] = meanValueSigma6Slice
        dfSlice['Mean Grey Matter dsp sigma8'] = meanValueSigma8Slice

        name = 'SimulateDataWith'+nroModel+conjunto+'SubjectSlices5%.xlsx'

        dfSlice.to_excel(pathSaveResults+'/'+name)

    allModelsCovAntes.append(np.mean(covAntesTodos))
    allModelsCovDsp.append(np.mean(covDspTodos))
    allModelsCovDspSigma4.append(np.mean(covTodosDspSigma4))
    allModelsCovDspSigma6.append(np.mean(covTodosDspSigma6))
    allModelsCovDspSigma8.append(np.mean(covTodosDspSigma8))

    allModelsCrcAntesTodos.append(np.mean(crcAntesTodos))
    allModelsCrcDsp.append(np.mean(crcDspTodos))
    allModelsCrcDspSigma4.append(np.mean(crcTodosDspSigma4))
    allModelsCrcDspSigma6.append(np.mean(crcTodosDspSigma6))
    allModelsCrcDspSigma8.append(np.mean(crcTodosDspSigma8))

    allModelsMeanValueAntes.append(np.mean(meanValueAntes))
    allModelsMeanValueDsp.append(np.mean(meanValueDsp))
    allModelsMeanValueSigma4.append(np.mean(meanValueSigma4))
    allModelsMeanValueSigma6.append(np.mean(meanValueSigma6))
    allModelsMeanValueSigma8.append(np.mean(meanValueSigma8))

    allModelsMeanValueAntesWhiteMatter.append(np.mean(meanValueAntesWhiteMatter))
    allModelsMeanValueDspWhiteMatter.append(np.mean(meanValueDspWhiteMatter))
    allModelsMeanValueSigma4WhiteMatter.append(np.mean(meanValueSigma4WhiteMatter))
    allModelsMeanValueSigma6WhiteMatter.append(np.mean(meanValueSigma6WhiteMatter))
    allModelsMeanValueSigma8WhiteMatter.append(np.mean(meanValueSigma8WhiteMatter))

    allModelsMseValueAntes.append(np.mean(mseValueAntes))
    allModelsMseValueDsp.append(np.mean(mseValueDsp))
    allModelsMseValueSigma4.append(np.mean(mseValueSigma4))
    allModelsMseValueSigma6.append(np.mean(mseValueSigma6))
    allModelsMseValueSigma8.append(np.mean(mseValueSigma8))

    allModelsMseGreyMatterAntes.append(np.mean(mseGreyMatterAntes))
    allModelsMseGreyMatterDsp.append(np.mean(mseGreyMatterDsp))
    allModelsMseGreyMatterSigma4.append(np.mean(mseGreyMatterSigma4))
    allModelsMseGreyMatterSigma6.append(np.mean(mseGreyMatterSigma6))
    allModelsMseGreyMatterSigma8.append(np.mean(mseGreyMatterSigma8))

    allModelsMseWhiteMatterAntes.append(np.mean(mseWhiteMatterAntes))
    allModelsMseWhiteMatterDsp.append(np.mean(mseWhiteMatterDsp))
    allModelsMseWhiteMatterSigma4.append(np.mean(mseWhiteMatterSigma4))
    allModelsMseWhiteMatterSigma6.append(np.mean(mseWhiteMatterSigma6))
    allModelsMseWhiteMatterSigma8.append(np.mean(mseWhiteMatterSigma8))

    allModelsStdValueAntes.append(np.mean(stdValueAntes))
    allModelsStdValueDsp.append(np.mean(stdValueDsp))
    allModelsStdValueSigma4.append(np.mean(stdValueSigma4))
    allModelsStdValueSigma6.append(np.mean(stdValueSigma6))
    allModelsStdValueSigma8.append(np.mean(stdValueSigma8))

    #x = np.arange(0, len(allModelsCovDsp))
    #y1 = allModelsCovDsp
    #y2 = allModelsCovDspSigma4
    #y3 = allModelsCovDspSigma6
    #y4 = allModelsCovDspSigma8
    #plt.plot(x, y1, label = 'COV dsp')
    #plt.plot(x, y2, label = 'COV sigma 4')
    #plt.plot(x, y3, label = 'COV sigma 6')
    #plt.plot(x, y4, label = 'COV sigma 8')
    #plt.legend(loc="upper left")
    #plt.title('COV')
    #plt.draw()
    #plt.pause(0.0001)

    if ((len(allModelsCovAntes))%5==0):
        dfAllModels = pd.DataFrame()

        dfAllModels['COV antes'] = allModelsCovAntes
        dfAllModels['COV dsp'] = allModelsCovDsp
        dfAllModels['COV dsp Filtro Sigma 4'] = allModelsCovDspSigma4
        dfAllModels['COV dsp Filtro Sigma 6'] = allModelsCovDspSigma6
        dfAllModels['COV dsp Filtro Sigma 8'] = allModelsCovDspSigma8

        dfAllModels['CRC antes'] = allModelsCrcAntesTodos
        dfAllModels['CRC dsp'] = allModelsCrcDsp
        dfAllModels['CRC dsp Filtro Sigma 4'] = allModelsCrcDspSigma4
        dfAllModels['CRC dsp Filtro Sigma 6'] = allModelsCrcDspSigma6
        dfAllModels['CRC dsp Filtro Sigma 8'] = allModelsCrcDspSigma8

        dfAllModels['Mean GM antes'] = allModelsMeanValueAntes
        dfAllModels['Mean GM dsp model'] = allModelsMeanValueDsp
        dfAllModels['Mean GM dsp sigma4'] = allModelsMeanValueSigma4
        dfAllModels['Mean GM dsp sigma6'] = allModelsMeanValueSigma6
        dfAllModels['Mean GM dsp sigma8'] = allModelsMeanValueSigma8

        dfAllModels['Mean WM antes'] = allModelsMeanValueAntesWhiteMatter
        dfAllModels['Mean WM dsp model'] = allModelsMeanValueDspWhiteMatter
        dfAllModels['Mean WM dsp sigma4'] = allModelsMeanValueSigma4WhiteMatter
        dfAllModels['Mean WM dsp sigma6'] = allModelsMeanValueSigma6WhiteMatter
        dfAllModels['Mean WM dsp sigma8'] = allModelsMeanValueSigma8WhiteMatter

        dfAllModels['MSE antes'] = allModelsMseValueAntes
        dfAllModels['MSE dsp model'] = allModelsMseValueDsp
        dfAllModels['MSE dsp sigma4'] = allModelsMseValueSigma4
        dfAllModels['MSE dsp sigma6'] = allModelsMseValueSigma6
        dfAllModels['MSE dsp sigma8'] = allModelsMseValueSigma8

        dfAllModels['MSE GM antes'] = allModelsMseGreyMatterAntes
        dfAllModels['MSE GM dsp model'] = allModelsMseGreyMatterDsp
        dfAllModels['MSE GM dsp sigma4'] = allModelsMseGreyMatterSigma4
        dfAllModels['MSE GM dsp sigma6'] = allModelsMseGreyMatterSigma6
        dfAllModels['MSE GM dsp sigma8'] = allModelsMseGreyMatterSigma8

        dfAllModels['MSE WM antes'] = allModelsMseWhiteMatterAntes
        dfAllModels['MSE WM dsp model'] = allModelsMseWhiteMatterDsp
        dfAllModels['MSE WM dsp sigma4'] = allModelsMseWhiteMatterSigma4
        dfAllModels['MSE WM dsp sigma6'] = allModelsMseWhiteMatterSigma6
        dfAllModels['MSE WM dsp sigma8'] = allModelsMseWhiteMatterSigma8

        dfAllModels['std GM antes'] = allModelsStdValueAntes
        dfAllModels['std GM dsp model'] = allModelsStdValueDsp
        dfAllModels['std GM dsp sigma4'] = allModelsStdValueSigma4
        dfAllModels['std GM dsp sigma6'] = allModelsStdValueSigma6
        dfAllModels['std GM dsp sigma8'] = allModelsStdValueSigma8

        dfAllModels['Model'] = modelName

        name = 'SimulateDataWithAll'+nameModel + conjunto + 'Subject5%Parcial.xlsx'

        dfAllModels.to_excel(pathSaveResults+'/'+name)


## Guardar en Excel
dfAllModels = pd.DataFrame()

dfAllModels['COV antes'] = allModelsCovAntes
dfAllModels['COV dsp'] = allModelsCovDsp
dfAllModels['COV dsp Filtro Sigma 4'] = allModelsCovDspSigma4
dfAllModels['COV dsp Filtro Sigma 6'] = allModelsCovDspSigma6
dfAllModels['COV dsp Filtro Sigma 8'] = allModelsCovDspSigma8

dfAllModels['CRC antes'] = allModelsCrcAntesTodos
dfAllModels['CRC dsp'] = allModelsCrcDsp
dfAllModels['CRC dsp Filtro Sigma 4'] = allModelsCrcDspSigma4
dfAllModels['CRC dsp Filtro Sigma 6'] = allModelsCrcDspSigma6
dfAllModels['CRC dsp Filtro Sigma 8'] = allModelsCrcDspSigma8

dfAllModels['Mean GM antes'] = allModelsMeanValueAntes
dfAllModels['Mean GM dsp model'] = allModelsMeanValueDsp
dfAllModels['Mean GM dsp sigma4'] = allModelsMeanValueSigma4
dfAllModels['Mean GM dsp sigma6'] = allModelsMeanValueSigma6
dfAllModels['Mean GM dsp sigma8'] = allModelsMeanValueSigma8

dfAllModels['Mean WM antes'] = allModelsMeanValueAntesWhiteMatter
dfAllModels['Mean WM dsp model'] = allModelsMeanValueDspWhiteMatter
dfAllModels['Mean WM dsp sigma4'] = allModelsMeanValueSigma4WhiteMatter
dfAllModels['Mean WM dsp sigma6'] = allModelsMeanValueSigma6WhiteMatter
dfAllModels['Mean WM dsp sigma8'] = allModelsMeanValueSigma8WhiteMatter

dfAllModels['MSE antes'] = allModelsMseValueAntes
dfAllModels['MSE dsp model'] = allModelsMseValueDsp
dfAllModels['MSE dsp sigma4'] = allModelsMseValueSigma4
dfAllModels['MSE dsp sigma6'] = allModelsMseValueSigma6
dfAllModels['MSE dsp sigma8'] = allModelsMseValueSigma8

dfAllModels['MSE GM antes'] = allModelsMseGreyMatterAntes
dfAllModels['MSE GM dsp model'] = allModelsMseGreyMatterDsp
dfAllModels['MSE GM dsp sigma4'] = allModelsMseGreyMatterSigma4
dfAllModels['MSE GM dsp sigma6'] = allModelsMseGreyMatterSigma6
dfAllModels['MSE GM dsp sigma8'] = allModelsMseGreyMatterSigma8

dfAllModels['MSE WM antes'] = allModelsMseWhiteMatterAntes
dfAllModels['MSE WM dsp model'] = allModelsMseWhiteMatterDsp
dfAllModels['MSE WM dsp sigma4'] = allModelsMseWhiteMatterSigma4
dfAllModels['MSE WM dsp sigma6'] = allModelsMseWhiteMatterSigma6
dfAllModels['MSE WM dsp sigma8'] = allModelsMseWhiteMatterSigma8

dfAllModels['std GM antes'] = allModelsStdValueAntes
dfAllModels['std GM dsp model'] = allModelsStdValueDsp
dfAllModels['std GM dsp sigma4'] = allModelsStdValueSigma4
dfAllModels['std GM dsp sigma6'] = allModelsStdValueSigma6
dfAllModels['std GM dsp sigma8'] = allModelsStdValueSigma8

dfAllModels['Model'] = modelName

name = 'SimulateDataWithAllModels'+nameModel+conjunto+'Subject5%.xlsx'

dfAllModels.to_excel(pathSaveResults+'/'+name)
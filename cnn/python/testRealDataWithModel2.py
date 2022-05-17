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
from utils import covValue
from utils import crcValue

import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

modelDT2 = Unet()
modelDT2.load_state_dict(torch.load('bestModelDataSet2_17'))

groundTruth_nii = sitk.ReadImage('./PET100.nii')
groundTruth_nii = sitk.GetArrayFromImage(groundTruth_nii)
groundTruth_nii = reshapeDataSet(groundTruth_nii)
groundTruth_np = torch.from_numpy(groundTruth_nii)

mseAntesTodos = []
mseDspTodos = []

idxSlices = []

# ---------------------------- MASCARAS ----------------------------------------------- #

whiteMatterMask_nii = sitk.ReadImage('./maskWhite.nii')
whiteMatterMask = sitk.GetArrayFromImage(whiteMatterMask_nii)
whiteMatterMask = reshapeDataSet(whiteMatterMask)
whiteMatterMask = torch.from_numpy(whiteMatterMask)

greyMatterMask_nii = sitk.ReadImage('./maskGrey.nii')
greyMatterMask = sitk.GetArrayFromImage(greyMatterMask_nii)
greyMatterMask = reshapeDataSet(greyMatterMask)
greyMatterMask = torch.from_numpy(greyMatterMask)


# ------------------------------------------------------------------------------------- #
noisyImages100_nii = sitk.ReadImage('./PET100.nii')
noisyImages100 = sitk.GetArrayFromImage(noisyImages100_nii)

noisyImages100 = reshapeDataSet(noisyImages100)
inputsRealData100 = torch.from_numpy(noisyImages100)

# PET al 100%
inputsRealData100Np = []
outRealData100Np = []
groundTruthTestRealData100Np = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []

covAntesTodos = []
covDspTodos = []

crcAntesTodos = []
crcDspTodos = []

print('------------------------------------PET 100------------------------------------------------')

# for idx in range(0,len(inputsRealData100)):
for idx in [80,81]:
    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData100[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData100[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData100Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)

    materiaGris = greyMatterMask[idx] * outModel
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)

    outRealData100Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData100Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)

# ------------------------------------------------------------------------------------- #
# PET al 50%

noisyImages50_nii = sitk.ReadImage('./PET50.nii')
noisyImages50 = sitk.GetArrayFromImage(noisyImages50_nii)

noisyImages50 = reshapeDataSet(noisyImages50)
inputsRealData50 = torch.from_numpy(noisyImages50)


inputsRealData50Np = []
outRealData50Np = []
groundTruthTestRealData50Np = []

print('-----------------------------------PET 50-------------------------------------------------')
for idx in range(0,len(inputsRealData50)):

    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData50[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData50[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData50Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)
    outRealData50Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData50Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)



# ------------------------------------------------------------------------------------- #
# PET al 25%

noisyImages25_nii = sitk.ReadImage('./PET25.nii')
noisyImages25 = sitk.GetArrayFromImage(noisyImages25_nii)

noisyImages25 = reshapeDataSet(noisyImages25)
inputsRealData25 = torch.from_numpy(noisyImages25)

cantImg = 3
cantIdx = (inputsRealData25.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsRealData25Np = []
outRealData25Np = []
groundTruthTestRealData25Np = []


print('-------------------------------------PET 25-----------------------------------------------')
for idx in range(0,len(inputsRealData25)):

    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData25[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData25[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData25Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)
    outRealData25Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData25Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)



# ------------------------------------------------------------------------------------- #
# PET al 10%

noisyImages10_nii = sitk.ReadImage('./PET10.nii')
noisyImages10 = sitk.GetArrayFromImage(noisyImages10_nii)

noisyImages10 = reshapeDataSet(noisyImages10)
inputsRealData10 = torch.from_numpy(noisyImages10)

inputsRealData10Np = []
outRealData10Np = []
groundTruthTestRealData10Np = []


print('--------------------------------------PET 10----------------------------------------------')
for idx in range(0,len(inputsRealData100)):

    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData10[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData10[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData10Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)
    outRealData10Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData10Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)


# ------------------------------------------------------------------------------------- #
# PET al 5%

noisyImages5_nii = sitk.ReadImage('./PET5.nii')
noisyImages5 = sitk.GetArrayFromImage(noisyImages5_nii)

noisyImages5 = reshapeDataSet(noisyImages5)
inputsRealData5 = torch.from_numpy(noisyImages5)

cantImg = 3
cantIdx = (inputsRealData5.shape[0])
ramdomIdx = np.random.randint(0, cantIdx, cantImg).tolist()

inputsRealData5Np = []
outRealData5Np = []
groundTruthTestRealData5Np = []


print('-------------------------------------PET 5-----------------------------------------------')
for idx in range(0,len(inputsRealData5)):

    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData5[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData5[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData5Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)
    outRealData5Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData5Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)


# ------------------------------------------------------------------------------------- #
# PET al 1%

noisyImages1_nii = sitk.ReadImage('./PET1.nii')
noisyImages1 = sitk.GetArrayFromImage(noisyImages1_nii)

noisyImages1 = reshapeDataSet(noisyImages1)
inputsRealData1 = torch.from_numpy(noisyImages1)


inputsRealData1Np = []
outRealData1Np = []
groundTruthTestRealData1Np = []


print('-------------------------------------PET 1-----------------------------------------------')
for idx in range(0,len(inputsRealData1)):

    # Normalizo la entrada con el valor de la materia gris
    materiaGris = greyMatterMask[idx] * inputsRealData1[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    inputNorm = (inputsRealData1[idx] / meanMateriaGris) * 8.0

    covAntesTodos.append(covValue(inputNorm,greyMatterMask[idx]))
    crcAntesTodos.append(crcValue(inputNorm, greyMatterMask[idx], whiteMatterMask[idx]))

    inputsRealData1Np.append(torchToNp(inputNorm))

    outModel = testModelSlice(modelDT2,inputNorm)
    outModelNp = torchToNp(outModel)
    outRealData1Np.append(outModelNp[0,:,:,:])

    covDspTodos.append(covValue(outModel[0,:,:], greyMatterMask[idx]))
    crcDspTodos.append(crcValue(outModel[0,:,:], greyMatterMask[idx], whiteMatterMask[idx]))

    #Normalizo groundTruth
    materiaGris = greyMatterMask[idx] * groundTruth_np[idx]
    materiaGris = np.trim_zeros((torchToNp(materiaGris)).flatten())
    meanMateriaGris = np.mean(materiaGris)
    gtNorm = (groundTruth_np[idx] / meanMateriaGris) * 8.0
    groundTruthTestRealData1Np.append(torchToNp(gtNorm))

    out, antesGrey, dpsGrey = getTestOneModelOneSlices(inputNorm, outModel[0,:,:,:], gtNorm,
                                                       mseGrey='True', maskGrey=greyMatterMask[idx])

    out, antesWhite, dpsWhite = getTestOneModelOneSlices( inputNorm,outModel[0,:,:,:], gtNorm,
                                                       mseWhite='True', maskWhite=whiteMatterMask[idx])


    idxSlices.append(idx)
    mseAntesGrey.append(antesGrey )
    mseDspGrey.append(dpsGrey )

    mseAntesWhite.append(antesWhite)
    mseDspWhite.append(dpsWhite)

## EXCEL
df = pd.DataFrame()

pet = np.concatenate((np.ones([(len(inputsRealData100)), 1]) * 100 ,np.ones([(len(inputsRealData50)), 1]) * 50 ,np.ones([(len(inputsRealData25)), 1]) * 25
                      ,np.ones([(len(inputsRealData10)), 1]) * 10, np.ones([(len(inputsRealData5)), 1]) * 5 ,np.ones([(len(inputsRealData1)), 1]) * 1))

df['Pet'] = pet
df['Slice'] = idxSlices

df['MSE Grey antes'] = mseAntesGrey
df['MSE Grey dsp'] = mseDspGrey

df['MSE White antes'] = mseAntesWhite
df['MSE White dsp'] = mseDspWhite

df['COV antes'] = covAntesTodos
df['COV dsp'] = covDspTodos

df['CRC antes'] = crcAntesTodos
df['CRC dsp'] = crcDspTodos


df.to_excel('RealDataWithModel2.xlsx')

image = sitk.GetImageFromArray(np.array(inputsRealData1Np))
sitk.WriteImage(image,'InputsRealData1Norm.nii' )

image = sitk.GetImageFromArray(np.array(outRealData1Np))
sitk.WriteImage(image,'outRealData1Norm.nii' )

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData1Np))
sitk.WriteImage(image,'groundTruthTestRealData1Norm.nii' )

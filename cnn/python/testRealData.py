import torch
import torchvision
import skimage

#from unetM import Unet
from unet import UnetWithResidual
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

modelDT2 = UnetWithResidual(1, 1)
modelDT2.load_state_dict(torch.load('UnetWithResidual_MSE_lr5e-05_AlignTrue_norm_20220715_191324_27_best_fit', map_location=torch.device(device)))

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

outFilter100Sigma4 = []
outFilter100Sigma6 = []
crcSliceDspSigma4 = []
covSliceDspSigma4 = []
crcSliceDspSigma6 = []
covSliceDspSigma6 = []

mseAntesGrey = []
mseDspGrey = []

mseAntesWhite = []
mseDspWhite = []

covAntesSlice = []
covDspSlice = []

covAntesTodos = []
covDspTodos = []

crcAntesSlice = []
crcDspSlice = []

crcAntesTodos = []
crcDspTodos = []


meanValueAntes = []
meanValueDsp = []

covTodosDspSigma6 = []
covTodosDspSigma4 = []

crcTodosDspSigma6 = []
crcTodosDspSigma4 = []

for idx in range(0,len(inputsRealData100)):
    maxSliceInp = inputsRealData100[idx, :, :].max()
    inputsRealDataNorm = inputsRealData100[idx]/maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData100Np.append(torchToNp(inputsRealData100[idx]))
    groundTruthTestRealData100Np.append(torchToNp(groundTruth_np[idx]))
    outRealData100Np.append(outModelNp[0, :, :, :])

    outFilter100Sigma4.append(skimage.filters.gaussian(inputsRealData100[idx, :, :], sigma=(2/2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter100Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter100Sigma4[-1]), greyMatterMask[idx]))

    outFilter100Sigma6.append(skimage.filters.gaussian(inputsRealData100[idx, :, :], sigma=(3/2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter100Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter100Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData100[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData100[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData100Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData100Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)


covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData100Np)[:,0,:,:]),greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData100Np)[:,0,:,:]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData100Np)[:,0,:,:]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData100Np)[:,0,:,:]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter100Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter100Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter100Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter100Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(np.mean(np.trim_zeros(((np.array(inputsRealData100Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outRealData100Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData100Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData100.nii')

image = sitk.GetImageFromArray(np.array(outRealData100Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData100.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData100Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData100.nii')

image = sitk.GetImageFromArray(np.array(outFilter100Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData100.nii')

image = sitk.GetImageFromArray(np.array(outFilter100Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData100.nii')

# ------------------------------------------------------------------------------------- #
# PET al 50%

noisyImages50_nii = sitk.ReadImage('./PET50.nii')
noisyImages50 = sitk.GetArrayFromImage(noisyImages50_nii)

noisyImages50 = reshapeDataSet(noisyImages50)
inputsRealData50 = torch.from_numpy(noisyImages50)

outFilter50Sigma4 = []
outFilter50Sigma6 = []

inputsRealData50Np = []
outRealData50Np = []
groundTruthTestRealData50Np = []

print('------------------------------------PET 50------------------------------------------------')

for idx in range(0,len(inputsRealData50)):
    maxSliceInp = inputsRealData50[idx, :, :].max()
    inputsRealDataNorm = inputsRealData50[idx] / maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData50Np.append(torchToNp(inputsRealData50[idx]))
    groundTruthTestRealData50Np.append(torchToNp(groundTruth_np[idx]))
    outRealData50Np.append(outModelNp[0, :, :, :])

    outFilter50Sigma4.append(skimage.filters.gaussian(inputsRealData50[idx, :, :], sigma=(2 / 2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter50Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter50Sigma4[-1]), greyMatterMask[idx]))

    outFilter50Sigma6.append(skimage.filters.gaussian(inputsRealData50[idx, :, :], sigma=(3 / 2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter50Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter50Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData50[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData50[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData50Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData50Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)

covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData50Np)[:, 0, :, :]), greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData50Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData50Np)[:, 0, :, :]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData50Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter50Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter50Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter50Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter50Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(
    np.mean(np.trim_zeros(((np.array(inputsRealData50Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(
    np.mean(np.trim_zeros(((np.array(outRealData50Np))[:,0,:,:,:] * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData50Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData50.nii')

image = sitk.GetImageFromArray(np.array(outRealData50Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData50.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData50Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData50.nii')

image = sitk.GetImageFromArray(np.array(outFilter50Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData50.nii')

image = sitk.GetImageFromArray(np.array(outFilter50Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData50.nii')
# ------------------------------------------------------------------------------------- #
# PET al 25%

noisyImages25_nii = sitk.ReadImage('./PET25.nii')
noisyImages25 = sitk.GetArrayFromImage(noisyImages25_nii)

noisyImages25 = reshapeDataSet(noisyImages25)
inputsRealData25 = torch.from_numpy(noisyImages25)

outFilter25Sigma4 = []
outFilter25Sigma6 = []

inputsRealData25Np = []
outRealData25Np = []
groundTruthTestRealData25Np = []



print('------------------------------------PET 25------------------------------------------------')

for idx in range(0,len(inputsRealData25)):
    maxSliceInp = inputsRealData25[idx, :, :].max()
    inputsRealDataNorm = inputsRealData25[idx] / maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData25Np.append(torchToNp(inputsRealData25[idx]))
    groundTruthTestRealData25Np.append(torchToNp(groundTruth_np[idx]))
    outRealData25Np.append(outModelNp[0, :, :, :])

    outFilter25Sigma4.append(skimage.filters.gaussian(inputsRealData25[idx, :, :], sigma=(2 / 2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter25Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter25Sigma4[-1]), greyMatterMask[idx]))

    outFilter25Sigma6.append(skimage.filters.gaussian(inputsRealData25[idx, :, :], sigma=(3 / 2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter25Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter25Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData25[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData25[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData25Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData25Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)

covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData25Np)[:, 0, :, :]), greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData25Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData25Np)[:, 0, :, :]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData25Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter25Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter25Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter25Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter25Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(np.mean(np.trim_zeros(((np.array(inputsRealData25Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outRealData25Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData25Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData25.nii')

image = sitk.GetImageFromArray(np.array(outRealData25Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData25.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData25Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData25.nii')

image = sitk.GetImageFromArray(np.array(outFilter25Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData25.nii')

image = sitk.GetImageFromArray(np.array(outFilter25Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData25.nii')


# ------------------------------------------------------------------------------------- #
# PET al 10%

noisyImages10_nii = sitk.ReadImage('./PET10.nii')
noisyImages10 = sitk.GetArrayFromImage(noisyImages10_nii)

noisyImages10 = reshapeDataSet(noisyImages10)
inputsRealData10 = torch.from_numpy(noisyImages10)

outFilter10Sigma4 = []
outFilter10Sigma6 = []


inputsRealData10Np = []
outRealData10Np = []
groundTruthTestRealData10Np = []


print('------------------------------------PET 10------------------------------------------------')

for idx in range(0,len(inputsRealData10)):
    maxSliceInp = inputsRealData10[idx, :, :].max()
    inputsRealDataNorm = inputsRealData10[idx] / maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData10Np.append(torchToNp(inputsRealData10[idx]))
    groundTruthTestRealData10Np.append(torchToNp(groundTruth_np[idx]))
    outRealData10Np.append(outModelNp[0, :, :, :])

    outFilter10Sigma4.append(skimage.filters.gaussian(inputsRealData10[idx, :, :], sigma=(2 / 2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter10Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter10Sigma4[-1]), greyMatterMask[idx]))

    outFilter10Sigma6.append(skimage.filters.gaussian(inputsRealData10[idx, :, :], sigma=(3 / 2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter10Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter10Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData10[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData10[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData10Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData10Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)

covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData10Np)[:, 0, :, :]), greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData10Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData10Np)[:, 0, :, :]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData10Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter10Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter10Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter10Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter10Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(np.mean(np.trim_zeros(((np.array(inputsRealData10Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outRealData10Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData10Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData10.nii')

image = sitk.GetImageFromArray(np.array(outRealData10Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData10.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData10Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData10.nii')

image = sitk.GetImageFromArray(np.array(outFilter10Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData10.nii')

image = sitk.GetImageFromArray(np.array(outFilter10Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData10.nii')

# ------------------------------------------------------------------------------------- #
# PET al 5%

noisyImages5_nii = sitk.ReadImage('./PET5.nii')
noisyImages5 = sitk.GetArrayFromImage(noisyImages5_nii)

noisyImages5 = reshapeDataSet(noisyImages5)
inputsRealData5 = torch.from_numpy(noisyImages5)

outFilter5Sigma4 = []
outFilter5Sigma6 = []

inputsRealData5Np = []
outRealData5Np = []
groundTruthTestRealData5Np = []

print('------------------------------------PET 5------------------------------------------------')

for idx in range(0,len(inputsRealData5)):
    maxSliceInp = inputsRealData5[idx, :, :].max()
    inputsRealDataNorm = inputsRealData5[idx] / maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData5Np.append(torchToNp(inputsRealData5[idx]))
    groundTruthTestRealData5Np.append(torchToNp(groundTruth_np[idx]))
    outRealData5Np.append(outModelNp[0, :, :, :])

    outFilter5Sigma4.append(skimage.filters.gaussian(inputsRealData5[idx, :, :], sigma=(2 / 2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter5Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter5Sigma4[-1]), greyMatterMask[idx]))

    outFilter5Sigma6.append(skimage.filters.gaussian(inputsRealData5[idx, :, :], sigma=(3 / 2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter5Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter5Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData5[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData5[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData5Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData5Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)

covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData5Np)[:, 0, :, :]), greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData5Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData5Np)[:, 0, :, :]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData5Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter5Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter5Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter5Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter5Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(np.mean(np.trim_zeros(((np.array(inputsRealData5Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outRealData5Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData5Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData5.nii')

image = sitk.GetImageFromArray(np.array(outRealData5Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData5.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData5Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData5.nii')

image = sitk.GetImageFromArray(np.array(outFilter5Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData5.nii')

image = sitk.GetImageFromArray(np.array(outFilter5Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData5.nii')


# ------------------------------------------------------------------------------------- #
# PET al 1%

noisyImages1_nii = sitk.ReadImage('./PET1.nii')
noisyImages1 = sitk.GetArrayFromImage(noisyImages1_nii)

noisyImages1 = reshapeDataSet(noisyImages1)
inputsRealData1 = torch.from_numpy(noisyImages1)

outFilter1Sigma4 = []
outFilter1Sigma6 = []


inputsRealData1Np = []
outRealData1Np = []
groundTruthTestRealData1Np = []


print('------------------------------------PET 1------------------------------------------------')

for idx in range(0,len(inputsRealData1)):
    maxSliceInp = inputsRealData100[idx, :, :].max()
    inputsRealDataNorm = inputsRealData100[idx] / maxSliceInp
    inputsRealDataNorm = np.nan_to_num(inputsRealDataNorm)

    outModel = testModelSlice(modelDT2, torch.from_numpy(inputsRealDataNorm))

    outModelNp = torchToNp((outModel) * maxSliceInp)

    inputsRealData1Np.append(torchToNp(inputsRealData1[idx]))
    groundTruthTestRealData1Np.append(torchToNp(groundTruth_np[idx]))
    outRealData1Np.append(outModelNp[0, :, :, :])

    outFilter1Sigma4.append(skimage.filters.gaussian(inputsRealData1[idx, :, :], sigma=(2 / 2.35)))
    crcSliceDspSigma4.append(crcValue(torch.from_numpy(outFilter1Sigma4[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma4.append(covValue(torch.from_numpy(outFilter1Sigma4[-1]), greyMatterMask[idx]))

    outFilter1Sigma6.append(skimage.filters.gaussian(inputsRealData1[idx, :, :], sigma=(3 / 2.35)))
    crcSliceDspSigma6.append(crcValue(torch.from_numpy(outFilter1Sigma6[-1]), greyMatterMask[idx], whiteMatterMask[idx]))
    covSliceDspSigma6.append(covValue(torch.from_numpy(outFilter1Sigma6[-1]), greyMatterMask[idx]))

    covAntesSlice.append(covValue(inputsRealData1[idx], greyMatterMask[idx]))
    crcAntesSlice.append(crcValue(inputsRealData1[idx], greyMatterMask[idx], whiteMatterMask[idx]))

    covDspSlice.append(covValue(torch.from_numpy(np.array(outRealData1Np)[idx, 0, :, :]), greyMatterMask[idx]))
    crcDspSlice.append(crcValue(torch.from_numpy(np.array(outRealData1Np)[idx, 0, :, :]), greyMatterMask[idx], whiteMatterMask[idx]))

    idxSlices.append(idx)

covAntesTodos.append(covValue(torch.from_numpy(np.array(inputsRealData1Np)[:, 0, :, :]), greyMatterMask))
crcAntesTodos.append(crcValue(torch.from_numpy(np.array(inputsRealData1Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covDspTodos.append(covValue(torch.from_numpy(np.array(outRealData1Np)[:, 0, :, :]), greyMatterMask))
crcDspTodos.append(crcValue(torch.from_numpy(np.array(outRealData1Np)[:, 0, :, :]), greyMatterMask, whiteMatterMask))

covTodosDspSigma6.append(covValue(torch.from_numpy(np.array(outFilter1Sigma6)),greyMatterMask))
covTodosDspSigma4.append(covValue(torch.from_numpy(np.array(outFilter1Sigma4)),greyMatterMask))

crcTodosDspSigma6.append(crcValue(torch.from_numpy(np.array(outFilter1Sigma6)), greyMatterMask, whiteMatterMask))
crcTodosDspSigma4.append(crcValue(torch.from_numpy(np.array(outFilter1Sigma4)), greyMatterMask, whiteMatterMask))

meanValueAntes.append(np.mean(np.trim_zeros(((np.array(inputsRealData1Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))
meanValueDsp.append(np.mean(np.trim_zeros(((np.array(outRealData1Np)[:,0,:,:,:]) * torch.Tensor.numpy(greyMatterMask)).flatten())))

image = sitk.GetImageFromArray(np.array(inputsRealData1Np)[:,0,0,:,:])
sitk.WriteImage(image, 'inputsRealData1.nii')

image = sitk.GetImageFromArray(np.array(outRealData1Np)[:,0,0,:,:])
sitk.WriteImage(image, 'outRealData1.nii')

image = sitk.GetImageFromArray(np.array(groundTruthTestRealData1Np)[:,0,0,:,:])
sitk.WriteImage(image, 'groundTruthTestRealData1.nii')

image = sitk.GetImageFromArray(np.array(outFilter1Sigma4)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma4TestRealData1.nii')

image = sitk.GetImageFromArray(np.array(outFilter1Sigma6)[:,0,:,:])
sitk.WriteImage(image, 'filterSigma6TestRealData1.nii')


## Guardar en Excel
dfGlobal = pd.DataFrame()

pet = np.concatenate((np.ones([1, 1]) * 100 ,np.ones([1, 1]) * 50 ,np.ones([1, 1]) * 25
                      ,np.ones([1, 1]) * 10, np.ones([1, 1]) * 5 ,np.ones([1, 1]) * 1))

dfGlobal['COV antes'] = covAntesTodos
dfGlobal['COV dsp'] = covDspTodos
dfGlobal['COV dsp Filtro Sigma 4'] = covTodosDspSigma4
dfGlobal['COV dsp Filtro Sigma 6'] = covTodosDspSigma6

dfGlobal['CRC antes'] = crcAntesTodos
dfGlobal['CRC dsp'] = crcDspTodos
dfGlobal['CRC dsp Filtro Sigma 4'] = crcTodosDspSigma4
dfGlobal['CRC dsp Filtro Sigma 6'] = crcTodosDspSigma6

dfGlobal['MEAN GREY MATTER antes'] = meanValueAntes
dfGlobal['MEAN GREY MATTER dsp'] = meanValueDsp

dfGlobal['Pet'] = pet

dfGlobal.to_excel('RealDataWithModel3GlobalNew.xlsx')

## Guardar en Excel
dfSlice = pd.DataFrame()

pet = np.concatenate((np.ones([len(inputsRealData100), 1]) * 100 ,np.ones([len(inputsRealData50), 1]) * 50 ,np.ones([len(inputsRealData25), 1]) * 25
                      ,np.ones([len(inputsRealData10), 1]) * 10, np.ones([len(inputsRealData5), 1]) * 5 ,np.ones([len(inputsRealData1), 1]) * 1))

dfSlice['Slice'] = idxSlices

dfSlice['COV antes'] = covAntesSlice
dfSlice['COV dsp model'] = covDspSlice
dfSlice['COV dsp sigma4'] = covSliceDspSigma4
dfSlice['COV dsp sigma6'] = covSliceDspSigma6

dfSlice['CRC antes'] = crcAntesSlice
dfSlice['CRC dsp'] = crcDspSlice
dfSlice['CRC dsp sigma4'] = crcSliceDspSigma4
dfSlice['CRC dsp sigma6'] = crcSliceDspSigma6


dfSlice['Pet'] = pet

dfSlice.to_excel('RealDataWithModel3SlicesNew.xlsx')
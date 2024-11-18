import torch
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import os
import SimpleITK as sitk
import numpy as np
import math

from cnn.python.utils import mseValuePerSubject
from utils import reshapeDataSet
from utils import saveDataCsv
from utils import crcValuePerSubject
from utils import covValuePerSubject
from utils import meanPerSubject
from utils import stdPerSubject
from utils import RunModel

#from unet import UnetWithResidual5Layers
from unet3d import UnetWithResidual
from unet3d import Unet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

######## CONFIG ###########

dataset = 'dataset_1'

# Subject
batchSubjects = False
batchSubjectsSize = 1


saveFilterOutputAsNiftiImage = True
saveModelOutputAsNiftiImage = True

# pet dose
petDose = [1,5,10,25,50,100]

######### PATHS ############
path = os.getcwd()

# Data path
dataPath = '../../data/RealData/' + dataset + '/'

# model
nameModel = 'NUEVA_UnetResidual_MSE_lr1e-05_AlignTrue_MeanBrain'

modelsPath = '../../results/newContrast/3dRESULTS/' + nameModel + '/models/'
modelFilenames = os.listdir(modelsPath)

# Normalize
normMeanStd = False
normMean = False
normMax = False
normBrainMean = True
model = UnetWithResidual(1,1)
#model = Unet(1,1)

#summary(model,(1,256,256))

# Output
pathSaveResults = '../../results/newContrast/3dRESULTS/' + nameModel + '/'


# leo las imagenes
pathImages = dataPath+'/pet'
arrayPet = os.listdir(pathImages)

namesPet = []
petImages = []

meanGlobalSubjectNoisy = []
maxGlobalSubjectNoisy = []
noisyImagesArrayOrig = []

groundTruthSubject = []
meanSlicesWithoutZeros = np.zeros((6,127))
cont = 0

for element in arrayPet:
    pathPetElement = pathImages+'/'+element
    name, extension = os.path.splitext(element)
    namesPet.append(name)

    # read GroundTruth
    petImg = sitk.ReadImage(pathPetElement)
    voxelSize_mm = petImg.GetSpacing()
    pet = sitk.GetArrayFromImage(petImg)
    pet = reshapeDataSet(pet)

    name, extension = os.path.splitext(element)
    if extension == '.nii':
        name, extension2 = os.path.splitext(name)

        if name == '100':
            groundTruthSubject = pet

    petImages.append(pet)

petImages = np.array(petImages)
groundTruthSubject = np.array(groundTruthSubject).squeeze()

# ---------------------------- MASCARAS ----------------------------------------------- #

whiteMatterMask_nii = sitk.ReadImage(dataPath+'mask/maskWhite.nii')
whiteMatterMask = sitk.GetArrayFromImage(whiteMatterMask_nii)
whiteMatterMask = reshapeDataSet(whiteMatterMask)
whiteMaskArray = torch.from_numpy(whiteMatterMask)

greyMatterMask_nii = sitk.ReadImage(dataPath+'mask/maskGrey.nii')
greyMatterMask = sitk.GetArrayFromImage(greyMatterMask_nii)
greyMatterMask = reshapeDataSet(greyMatterMask)
greyMaskArray = torch.from_numpy(greyMatterMask)


#### Normalizacion ###

if normMeanStd == True:
    meanSubjectNoisy = np.mean(petImages, axis=(1,2,3,4))
    stdSubjectNoisy = np.std(petImages, axis=(1,2,3,4))
    stdSubjectNoisy = np.where(stdSubjectNoisy == 0, np.nan, stdSubjectNoisy)
    subjectNoisyNorm = (petImages- meanSubjectNoisy[:,None, None,None, None]) / stdSubjectNoisy[:,None, None,None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normMean == True:
    meanSubjectNoisy = np.mean(petImages, axis=(1,2,3,4))
    meanSubjectNoisy = np.where(meanSubjectNoisy == 0, np.nan, meanSubjectNoisy)
    subjectNoisyNorm = petImages / meanSubjectNoisy[:,None, None,None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normMax == True:
    maxSubjectNoisy = np.max(petImages, axis=(1,2,3,4))
    maxSubjectNoisy = np.where(maxSubjectNoisy == 0, np.nan, maxSubjectNoisy)
    subjectNoisyNorm = petImages / maxSubjectNoisy[:,None, None,None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normBrainMean == True:
    BrainMask = whiteMaskArray + greyMaskArray
    BrainMask = np.where(BrainMask != 0, 1, 0)

    BrainMask = np.tile(BrainMask, ((petImages.shape)[0], 1, 1, 1, 1))

    meanSliceTrainNoisy = (meanPerSubject(petImages[:, :, 0, :, :]* BrainMask[:, :, 0, :, :])).astype(np.float32)
    subjectNoisyNorm = petImages/ meanSliceTrainNoisy[:,None, None, None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)


if normMeanStd == False and normMean == False and normMax == False and normBrainMean == False:
    petImagesNorm = petImages


## CODE ##
# Input images
meanGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))
meanWhiteMatterInputImagePerSubject = np.zeros((len(arrayPet)))

covInputImagePerSubject = np.zeros((len(arrayPet)))
crcInputImagePerSubject = np.zeros((len(arrayPet)))
stdGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))
stdWhiteMatterInputImagePerSubject = np.zeros((len(arrayPet)))
mseInputImagePerSubject = np.zeros((len(arrayPet)))
mseGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))


# Input images + filter
filtersFWHM_mm = np.array([2,4,6,8,10,12])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0] # we use an isometric filter in voxels (using x dimension as voxel size, not ideal)
filtersStdDev_voxels = filtersFWHM_voxels/2.35
meanGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
meanWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
covFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
crcFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
mseFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
stdGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
stdWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
mseGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))

# Input images + models
allModelsMeanGMperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsMeanWMperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsCOVperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsCRCperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsMsePerSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsGreyMatterMsePerSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsStdGreyMatterPerSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsStdWhiteMatterPerSubject = np.zeros((len(modelFilenames), len(arrayPet)))

filterSub = []

whiteMaskSubject = torch.Tensor.numpy(whiteMaskArray.squeeze())
greyMaskSubject = torch.Tensor.numpy(greyMaskArray.squeeze())

for dose in range(0, len(petImages)):
    # Get images for one subject as a torch tensor:
    noisyImagesSubject = petImages[dose, :,:, :, :].squeeze()
    print('PET DOSE ', namesPet[dose])

    # calculo los filtros
    # los voy a guardar

    # METRICAS INPUT IMAGE
    mask = (noisyImagesSubject * greyMaskSubject)
    #meanGreyMatterInputImagePerSubject[dose] = np.mean(np.mean(np.mean((mask), axis = -1), axis=-1), axis=-1)
    meanGreyMatterInputImagePerSubject[dose] = meanPerSubject(mask)
    #mseGreyMatterInputImagePerSlice[dose] = mseValuePerSlice(noisyImagesSubject, groundTruthSubject, greyMaskSubject)
    mseGreyMatterInputImagePerSubject[dose] = mseValuePerSubject(noisyImagesSubject, groundTruthSubject,greyMaskSubject)
    mask = (noisyImagesSubject * whiteMaskSubject)
    meanWhiteMatterInputImagePerSubject[dose] = np.mean(np.mean(np.mean((mask), axis = -1), axis=-1), axis=-1)

    crcInputImagePerSubject[dose] = crcValuePerSubject(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSubject[dose] = covValuePerSubject(noisyImagesSubject, greyMaskSubject)

    # METRICAS FILTROS + INPUT IMAGE
    for fil in range(0, len(filtersStdDev_voxels)):
        filtStdDev_voxels = filtersStdDev_voxels[fil]
        # Now doing a 2D filter to match the 2D processing of the UNET:
        filter = (skimage.filters.gaussian(noisyImagesSubject, sigma=(filtStdDev_voxels,filtStdDev_voxels,filtStdDev_voxels))).squeeze()

        if saveFilterOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(filter))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Dose' + str(namesPet[dose]) +'_filter_'+str(fil)+'_'+dataset+'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        mask = (filter * greyMaskSubject)
        meanGreyMatterFilterPerSubject[fil,dose] = meanPerSubject(mask)
        #meanGreyMatterFilterPerSubject[fil,dose] = np.mean(np.mean(np.mean((mask), axis = -1), axis=-1), axis=-1)
        mseGreyMatterFilterPerSubject[fil, dose] = mseValuePerSubject(filter, groundTruthSubject, greyMaskSubject)
        mask= (filter * whiteMaskSubject)
        #meanWhiteMatterFilterPerSubject[fil,dose] = np.mean(np.mean(np.mean((mask), axis = -1), axis=-1), axis=-1)
        meanWhiteMatterFilterPerSubject[fil,dose] = meanPerSubject(mask)

        crcFilterPerSubject[fil, dose] = crcValuePerSubject(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSubject[fil, dose] = covValuePerSubject(filter, greyMaskSubject)

        #stdGreyMatterFilterPerSubject[fil, dose] = stdPerSubject(filter * greyMaskSubject)
        #stdWhiteMatterFilterPerSubject[fil, dose] = stdPerSubject(filter * whiteMaskSubject)

outModel = np.zeros((petImages.shape[1], 1,petImages.shape[3], petImages.shape[3]))
contModel = 0
modelName = []

for modelFilename in modelFilenames:
    model.load_state_dict(torch.load(modelsPath + modelFilename, map_location=torch.device(device)))
    modelName.append(modelFilename)

    print('Model',contModel+1)

    for dose in range(0, len(petImages)):
        # Get images for one subject as a torch tensor:
        noisyImagesSubject = petImagesNorm[dose, :, : , :]

        if batchSubjects:
            # Divido el dataSet
            numBatches = np.round(noisyImagesSubject.shape[0] / batchSubjectsSize).astype(int)
            # Run the model for all the slices:
            for i in range(numBatches):
                outModel[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :, :, :] = RunModel(model, torch.from_numpy(
                noisyImagesSubject[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :, :, :])).detach().numpy()
                ndaOutputModel = outModel
        else:
            changeNoisyImagesSubject = np.transpose(noisyImagesSubject, (1, 0, 2, 3))
            outModel = RunModel(model, torch.from_numpy(changeNoisyImagesSubject).unsqueeze(1)).detach().numpy()
            ndaOutputModel = outModel
            # Convert it into numpy:

        if normMeanStd:
            ndaOutputModel = ndaOutputModel * stdSubjectNoisy[dose,None, None, None, None] + meanSubjectNoisy[dose,None, None, None, None]

        if normMean:
            ndaOutputModel = ndaOutputModel * meanSubjectNoisy[dose,None, None, None, None]

        if normMax:
            ndaOutputModel = ndaOutputModel * maxSubjectNoisy[dose,None, None, None, None]

        if normBrainMean:
            ndaOutputModel = ndaOutputModel *meanSliceTrainNoisy[dose,None, None, None, None]

        ndaOutputModel = ndaOutputModel.squeeze() # Remove the channel dimension

        if saveModelOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(ndaOutputModel))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Dose' + str(namesPet[dose]) +'_OutModel_'+modelName[-1]+'_'+dataset +'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        # Compute metrics for each slice:
        greyMaskedImage = (ndaOutputModel * greyMaskSubject)
        whiteMaskedImage = (ndaOutputModel * whiteMaskSubject)


        # Compute metrics for all subject :
        allModelsCRCperSubject[contModel,dose] = crcValuePerSubject(ndaOutputModel,greyMaskSubject,whiteMaskSubject)
        allModelsCOVperSubject[contModel,dose] = covValuePerSubject(ndaOutputModel, greyMaskSubject)
        #allModelsMeanGMperSubject[contModel, dose] = np.mean(np.mean(np.mean((ndaOutputModel* greyMaskSubject), axis = -1), axis=-1), axis=-1)
        #allModelsMeanWMperSubject[contModel, dose] = np.mean(np.mean(np.mean((ndaOutputModel* whiteMaskSubject), axis = -1), axis=-1), axis=-1)
        allModelsMeanGMperSubject[contModel, dose] = meanPerSubject(ndaOutputModel * greyMaskSubject)
        allModelsMeanWMperSubject[contModel, dose] = meanPerSubject(ndaOutputModel* whiteMaskSubject)
        allModelsStdGreyMatterPerSubject[contModel, dose] = stdPerSubject(ndaOutputModel * greyMaskSubject)
        allModelsStdWhiteMatterPerSubject[contModel, dose] = stdPerSubject(ndaOutputModel * whiteMaskSubject)
        allModelsGreyMatterMsePerSubject[contModel, dose] = mseValuePerSubject(ndaOutputModel, groundTruthSubject,greyMaskSubject)

        contModel = 0


saveDataCsv(meanGreyMatterInputImagePerSubject.T, 'meanInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcInputImagePerSubject.T, 'crcInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covInputImagePerSubject.T, 'covInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterInputImagePerSubject.T, 'mseInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSubject[1].T, 'meanFilter4mmImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSubject[1].T, 'crcFilter4mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSubject[1].T, 'covFilter4mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterFilterPerSubject[1].T, 'mseFilter4mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSubject[2].T, 'meanFilter6mmImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSubject[2].T, 'crcFilter6mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSubject[2].T, 'covFilter6mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterFilterPerSubject[2].T, 'mseFilter6mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSubject[3].T, 'meanFilter8mmImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSubject[3].T, 'crcFilter8mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSubject[3].T, 'covFilter8mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterFilterPerSubject[3].T, 'mseFilter8mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSubject[4].T, 'meanFilter10mmImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSubject[4].T, 'crcFilter10mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSubject[4].T, 'covFilter10mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterFilterPerSubject[4].T, 'mseFilter10mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSubject[5].T, 'meanFilter12mmImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSubject[5].T, 'crcFilter12mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSubject[5].T, 'covFilter12mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(mseGreyMatterFilterPerSubject[5].T, 'mseFilter12mm_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(allModelsMeanGMperSubject.squeeze().T, 'meanOutputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(allModelsCRCperSubject.squeeze().T, 'crcOutputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(allModelsCOVperSubject.squeeze().T, 'covOutput_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(allModelsGreyMatterMsePerSubject.squeeze().T, 'mseOutput_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
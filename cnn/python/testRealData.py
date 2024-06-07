import torch
import skimage
#from torchsummary import summary

from unet import Unet
from unet import UnetWithResidual
from utils import mseValuePerSlice
from utils import mseValuePerSubject
from utils import showDataPlot
from utils import showPlotGlobalData
from utils import reshapeDataSet
from utils import covValuePerSlice
from utils import crcValuePerSlice
from utils import crcValuePerSubject
from utils import covValuePerSubject
from utils import meanPerSlice
from utils import meanPerSubject
from utils import stdPerSubject
from utils import RunModel
from utils import saveDataCsv
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

######## CONFIG ###########

learning_rate=0.00005

dataset = 'dataset_1'

# Subject
batchSubjects = True
batchSubjectsSize = 20


saveFilterOutputAsNiftiImage = True
saveModelOutputAsNiftiImage = True

# pet dose
petDose = [1,5,10,25,50,100]

######### PATHS ############
path = os.getcwd()

# Data path
dataPath = '../../data/RealData/' + dataset + '/'

# model
nameModel = 'UnetResidual_MSE_lr5e-05_AlignTrue_randomScaleNew'.format(learning_rate)

modelsPath = '../../results/' + nameModel + '/models/'
modelFilenames = os.listdir(modelsPath)

# Normalize
normSliceMeanStd = False
normMeanSlice = False
normMaxSlice = False
model = UnetWithResidual(1,1)
#model = Unet(1,1)

#summary(model,(1,256,256))

# Output
pathSaveResults = '../../results/' + nameModel + '/'


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

if normSliceMeanStd == True:
    meanSubjectNoisy = np.mean(np.mean(petImages, axis=-1), axis=-1)
    stdSubjectNoisy = np.std(np.std(petImages, axis=-1), axis=-1)
    stdSubjectNoisy = np.where(stdSubjectNoisy == 0, np.nan, stdSubjectNoisy)
    subjectNoisyNorm = (petImages- meanSubjectNoisy[:,:, None, None]) / stdSubjectNoisy[:,:, None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normMeanSlice == True:
    meanSubjectNoisy = np.mean(np.mean(petImages, axis=-1), axis=-1)
    meanSubjectNoisy = np.where(meanSubjectNoisy == 0, np.nan, meanSubjectNoisy)
    subjectNoisyNorm = petImages / meanSubjectNoisy[:,:, None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normMaxSlice == True:
    maxSubjectNoisy = np.max(np.max(petImages, axis=-1), axis=-1)
    maxSubjectNoisy = np.where(maxSubjectNoisy == 0, np.nan, maxSubjectNoisy)
    subjectNoisyNorm = petImages / maxSubjectNoisy[:,:, None, None]
    petImagesNorm = np.nan_to_num(subjectNoisyNorm)

if normSliceMeanStd == False and normMeanSlice == False and normMaxSlice == False:
    petImagesNorm = petImages

# ---------------------------- MASCARAS ----------------------------------------------- #

whiteMatterMask_nii = sitk.ReadImage(dataPath+'mask/maskWhite.nii')
whiteMatterMask = sitk.GetArrayFromImage(whiteMatterMask_nii)
whiteMatterMask = reshapeDataSet(whiteMatterMask)
whiteMaskArray = torch.from_numpy(whiteMatterMask)

greyMatterMask_nii = sitk.ReadImage(dataPath+'mask/maskGrey.nii')
greyMatterMask = sitk.GetArrayFromImage(greyMatterMask_nii)
greyMatterMask = reshapeDataSet(greyMatterMask)
greyMaskArray = torch.from_numpy(greyMatterMask)

## CODE ##
# Input images
meanGreyMatterInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
meanWhiteMatterInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
meanGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))
meanWhiteMatterInputImagePerSubject = np.zeros((len(arrayPet)))

covInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
covInputImagePerSubject = np.zeros((len(arrayPet)))
stdGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))
stdWhiteMatterInputImagePerSubject = np.zeros((len(arrayPet)))
mseInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
mseInputImagePerSubject = np.zeros((len(arrayPet)))
mseGreyMatterInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
mseWhiteMatterInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
mseGreyMatterInputImagePerSubject = np.zeros((len(arrayPet)))
mseWhiteMatterInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))

crcInputImagePerSlice = np.zeros((len(arrayPet), petImages.shape[1]))
crcInputImagePerSubject = np.zeros((len(arrayPet)))


# Input images + filter
filtersFWHM_mm = np.array([2,4,6,8,10,12])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0] # we use an isometric filter in voxels (using x dimension as voxel size, not ideal)
filtersStdDev_voxels = filtersFWHM_voxels/2.35
meanGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
meanWhiteMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
meanGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
meanWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
covFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
crcFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
mseFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
mseGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),len(arrayPet), petImages.shape[1]))
covFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
crcFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
mseFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
stdGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
stdWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))
mseGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),len(arrayPet)))

# Input images + models
allModelsCrc = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
allModelsCov = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
allModelsMeanGM = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
allModelsMeanWM = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
allModelsMeanGMperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsMeanWMperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsCOVperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsCRCperSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsMsePerSlice = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
allModelsMsePerSubject = np.zeros((len(modelFilenames), len(arrayPet)))
allModelsGreyMatterMsePerSlice = np.zeros((len(modelFilenames), len(arrayPet), petImages.shape[1]))
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
    meanGreyMatterInputImagePerSlice[dose, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    mseGreyMatterInputImagePerSlice[dose, :] = mseValuePerSlice(noisyImagesSubject, groundTruthSubject, greyMaskSubject)
    #mseGreyMatterInputImagePerSubject[dose] = mseValuePerSubject(noisyImagesSubject, groundTruthSubject,greyMaskSubject)
    mask = (noisyImagesSubject * whiteMaskSubject)
    meanWhiteMatterInputImagePerSlice[dose, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    crcInputImagePerSlice[dose, :] = crcValuePerSlice(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSlice[dose, :] = covValuePerSlice(noisyImagesSubject, greyMaskSubject)
    crcInputImagePerSubject[dose] = crcValuePerSubject(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSubject[dose] = covValuePerSubject(noisyImagesSubject, greyMaskSubject)

    crcInputImageGlobal = np.mean(crcInputImagePerSubject[:])
    covInputImageGlobal = np.mean(covInputImagePerSubject[:])

    meanWhiteMatterInputImagePerSubject[dose] = meanPerSubject(meanWhiteMatterInputImagePerSlice[dose, :])
    meanGreyMatterInputImagePerSubject[dose] = meanPerSubject(meanGreyMatterInputImagePerSlice[dose, :])
    stdGreyMatterInputImagePerSubject[dose] = stdPerSubject(noisyImagesSubject * greyMaskSubject)
    stdWhiteMatterInputImagePerSubject[dose] = stdPerSubject(noisyImagesSubject * whiteMaskSubject)

    # METRICAS FILTROS + INPUT IMAGE
    for fil in range(0, len(filtersStdDev_voxels)):
        filtStdDev_voxels = filtersStdDev_voxels[fil]
        # Now doing a 2D filter to match the 2D processing of the UNET:
        filter = (skimage.filters.gaussian(noisyImagesSubject, sigma=(0,filtStdDev_voxels,filtStdDev_voxels))).squeeze()

        if saveFilterOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(filter))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Dose' + str(namesPet[dose]) +'_filter_'+str(fil)+'_'+dataset+'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        mask = (filter * greyMaskSubject)
        meanGreyMatterFilterPerSlice[fil,dose,:] = meanPerSlice((mask.reshape(mask.shape[0], -1)))
        mseGreyMatterFilterPerSlice[fil, dose, :] = mseValuePerSlice(filter, groundTruthSubject, greyMaskSubject)
        #mseGreyMatterFilterPerSubject[fil, dose] = mseValuePerSubject(filter, groundTruthSubject, greyMaskSubject)
        mask= (filter * whiteMaskSubject)
        meanWhiteMatterFilterPerSlice[fil,dose, :] = meanPerSlice((mask.reshape(mask.shape[0], -1)))

        crcFilterPerSlice[fil, dose, :] = crcValuePerSlice(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSlice[fil, dose, :] = covValuePerSlice(filter, greyMaskSubject)

        crcFilterPerSubject[fil, dose] = crcValuePerSubject(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSubject[fil, dose] = covValuePerSubject(filter, greyMaskSubject)
        meanGreyMatterFilterPerSubject[fil, dose] = meanPerSubject(meanGreyMatterFilterPerSlice[fil, dose, :])
        meanWhiteMatterFilterPerSubject[fil, dose] = meanPerSubject(meanWhiteMatterFilterPerSlice[fil, dose, :])
        stdGreyMatterFilterPerSubject[fil, dose] = stdPerSubject(filter * greyMaskSubject)
        stdWhiteMatterFilterPerSubject[fil, dose] = stdPerSubject(filter * whiteMaskSubject)

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
            outModel = RunModel(model, torch.from_numpy(noisyImagesSubject))
            # Convert it into numpy:
            ndaOutputModel = outModel.detach().numpy()

        if normSliceMeanStd:
            ndaOutputModel = ndaOutputModel * stdSubjectNoisy[dose,:, None, None] + meanSubjectNoisy[dose,:, None, None]

        if normMeanSlice:
            ndaOutputModel = ndaOutputModel * meanSubjectNoisy[dose,:, None, None]

        if normMaxSlice:
            ndaOutputModel = ndaOutputModel * maxSubjectNoisy[dose,:, None, None]

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

        # Compute metrics for each subject all slices:
        allModelsMeanGM[contModel,dose,:] = meanPerSlice((greyMaskedImage.reshape(greyMaskedImage.shape[0], -1)))
        allModelsMeanWM[contModel, dose, :] = meanPerSlice((whiteMaskedImage.reshape(whiteMaskedImage.shape[0], -1)))
        allModelsCrc[contModel,dose,:] = crcValuePerSlice(ndaOutputModel, greyMaskSubject, whiteMaskSubject)
        allModelsCov[contModel,dose,:] = covValuePerSlice(ndaOutputModel, greyMaskSubject)

        # Compute metrics for all subject :
        allModelsCRCperSubject[contModel,dose] = crcValuePerSubject(ndaOutputModel,greyMaskSubject,whiteMaskSubject)
        allModelsCOVperSubject[contModel,dose] = covValuePerSubject(ndaOutputModel, greyMaskSubject)
        allModelsMeanGMperSubject[contModel, dose] = meanPerSubject(allModelsMeanGM[contModel, dose, :])
        allModelsMeanWMperSubject[contModel, dose] = meanPerSubject(allModelsMeanWM[contModel, dose, :])
        allModelsStdGreyMatterPerSubject[contModel, dose] = stdPerSubject(ndaOutputModel * greyMaskSubject)
        allModelsStdWhiteMatterPerSubject[contModel, dose] = stdPerSubject(ndaOutputModel * whiteMaskSubject)
        #allModelsGreyMatterMsePerSubject[contModel, dose] = mseValuePerSubject(ndaOutputModel, groundTruthSubject,greyMaskSubject)

        contModel = 0

        meanGreyMatterFilterPerSlice[fil,dose,:] = meanPerSlice((mask.reshape(mask.shape[0], -1)))


saveDataCsv(meanGreyMatterInputImagePerSlice.T, 'meanInputImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcInputImagePerSlice.T, 'crcInputImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covInputImagePerSlice.T, 'covInputPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSlice[1,:,:].T, 'meanFilter2mmImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSlice[1,:,:].T, 'crcFilter4mmPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSlice[1,:,:].T, 'covFilterPer4mmSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSlice[2,:,:].T, 'meanFilter6mmImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSlice[2,:,:].T, 'crcFilter6mmPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSlice[2,:,:].T, 'covFilterPer6mmSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSlice[3,:,:].T, 'meanFilter8mmImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSlice[3,:,:].T, 'crcFilter8mmPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSlice[3,:,:].T, 'covFilterPer8mmSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSlice[4,:,:].T, 'meanFilter10mmImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSlice[4,:,:].T, 'crcFilter10mmPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSlice[4,:,:].T, 'covFilterPer10mmSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(meanGreyMatterFilterPerSlice[5,:,:].T, 'meanFilter12mmImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(crcFilterPerSlice[5,:,:].T, 'crcFilter12mmPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(covFilterPerSlice[5,:,:].T, 'covFilterPer12mmSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)

saveDataCsv(allModelsMeanGM.squeeze().T, 'meanOutputImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(allModelsCrc.squeeze().T, 'crcOutputImagePerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
saveDataCsv(allModelsCov.squeeze().T, 'covOutputPerSlice_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)


# saveDataCsv(crcFilterPerSubject, 'crcFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(covFilterPerSubject, 'covFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(crcInputImagePerSubject, 'crcInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(covInputImagePerSubject, 'covInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsCOVperSubject, 'covModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsCRCperSubject, 'crcModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsGreyMatterMsePerSubject, 'mseModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
#
# saveDataCsv(meanGreyMatterFilterPerSubject, 'meanGreyMatterFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(meanWhiteMatterFilterPerSubject, 'meanWhiteMatterFilterImage_RealData_'+dataset+'_' + nameModel + '.csv', pathSaveResults)
# saveDataCsv(stdGreyMatterFilterPerSubject, 'stdGreyMatterFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(stdWhiteMatterFilterPerSubject, 'stdWhiteMatterFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(mseGreyMatterFilterPerSubject, 'mseGreyMatterFilterImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
#
# saveDataCsv(meanGreyMatterInputImagePerSubject, 'meanGreyMatterInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(meanWhiteMatterInputImagePerSubject, 'meanWhiteMatterInputImage_RealData_'+dataset+'_' + nameModel + '.csv', pathSaveResults)
# saveDataCsv(stdGreyMatterInputImagePerSubject, 'stdGreyMatterInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(stdWhiteMatterInputImagePerSubject, 'stdWhiteMatterInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(mseGreyMatterInputImagePerSubject, 'mseGreyMatterInputImage_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
#
# saveDataCsv(allModelsMeanGMperSubject, 'meanGreyMatterModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsGreyMatterMsePerSubject, 'mseGreyMatterModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsMeanWMperSubject, 'meanWhiteMatterModels_RealData_' +dataset+'_'+ nameModel + '.csv', pathSaveResults)
# saveDataCsv(allModelsStdGreyMatterPerSubject, 'stdGreyMatterModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
# saveDataCsv(allModelsStdWhiteMatterPerSubject, 'stdWhiteMatterModels_RealData_'+dataset+'_'+nameModel+'.csv', pathSaveResults)
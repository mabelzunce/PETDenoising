import torch
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import os
import SimpleITK as sitk
import numpy as np
import math

#from unetM import Unet
from unet import Unet
from unet import UnetDe1a16Hasta512
from unet import UnetWithResidual
from unet import Unet512
from unet import UnetWithResidual5Layers
from utils import reshapeDataSet
from utils import mseValuePerSlice
from utils import meanPerSlice
from utils import mseValuePerSubject
from utils import showDataPlot
from utils import showPlotGlobalData
from utils import covValuePerSlice
from utils import crcValuePerSlice
from utils import RunModel
from utils import saveDataCsv

from sklearn.model_selection import train_test_split

learning_rate=0.00005
lowDose_perc = 5
fullDose = 100
epoch = 23
actScaleFactor = 100/lowDose_perc
allSubjects = [*range(1,21)]
validSubjects = [2, 4, 6, 8]
trainingSubjects = allSubjects
for i in validSubjects:
    trainingSubjects.remove(i)

calculateOnlyValidSubjectMetrics = True
calculateOnlyTrainingSubjectMetrics = False
calculateAllSubjectMetrics = False

# Subject

batchSubjects = True
batchSubjectsSize = 25
# Results visualization
showGlobalPlots = True
showPerfilSlices = True
showImageSub = False
# Save results
saveModelOutputAsNiftiImage = False
saveModelOutputAsNiftiImageOneSubject = True
saveFilterOutputAsNiftiImage = False
saveFilterOutputAsNiftiImageOneSubject = True
saveDataCSV = True
###########################

normSliceMeanStd = False
normBrainMean = True
normMeanGreyMatter = False
normMeanSlice = False
normMaxSlice = False
randonGaussian = False
randomScale = False

######################### CHECK DEVICE ######################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################

######### PATHS ############


path = os.getcwd()

# model
nameModel = 'PruebaMartin_Nueva_UnetResidual_MSE_lr1e-05_AlignTrue_MeanBrain'.format(learning_rate)

modelsPath = '../../results/newContrast/' + nameModel + '/models/'
readNormData = '../../results/' + nameModel +'/'

groundTruthSubdir = '100'
lowDoseSubdir = str(lowDose_perc)

# Output
pathSaveResults = '../../results/newContrast/' + nameModel + '/'

# Data path
#dataPath = '../../data/PSF/BrainWebSimulationsCompleteDataSet/'
dataPath = '../../data/BrainWebSimulations2D/'
#pathPhantoms = dataPath + '/Phantoms/'


########### CREATE MODEL ###########
model =UnetWithResidual(1,1)
#model= Unet512(1,1,32)
#model = UnetDe1a16Hasta512(1,1,16)
#model = UnetWithResidual(1,1)
#model = UnetWithResidual5Layers(1,1)


#model = Unet(1,1)


########## LIST OF MODELS ###############
modelFilenames = os.listdir(modelsPath)

########## PROCESSS DATA ###############

groundTruth = sitk.ReadImage(dataPath +'/testGroundTruth.nii')
voxelSize_mm = groundTruth.GetSpacing()
groundTruth = sitk.GetArrayFromImage(groundTruth)

noisyDataSet = sitk.ReadImage(dataPath +'/testNoisyDataset.nii')
noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

whiteMatter = sitk.ReadImage(dataPath +'/testWhiteMatterMask.nii')
whiteMatter = sitk.GetArrayFromImage(whiteMatter)

greyMatter  = sitk.ReadImage(dataPath +'/testGreyMatterMask.nii')
greyMatter  = sitk.GetArrayFromImage(greyMatter )

groundTruth = reshapeDataSet(groundTruth.squeeze())
noisyDataSet = reshapeDataSet(noisyDataSet.squeeze())
whiteMatter = reshapeDataSet(whiteMatter.squeeze())
greyMatter = reshapeDataSet(greyMatter.squeeze())


###

groundTruthDataSet = np.array(groundTruth).astype(np.float32)
noisyDataSetArray = np.array(noisyDataSet).astype(np.float32)
greyMatter = np.array(greyMatter).astype(np.float32)
whiteMatter = np.array(whiteMatter).astype(np.float32)

greyMatter = np.where(greyMatter != 0, 1, 0)
whiteMatter = np.where(whiteMatter != 0, 1, 0)

#noisyDataSet = np.expand_dims(noisyDataSetArray, axis=1)
calculateInputImages = False
calculateFiltersValues = False
calculateModelsValues = False
calculateGroundTruthValues = True

# Calculate grounf truth image metrics
if calculateGroundTruthValues:
    greyMaskGroundTruth = (groundTruthDataSet * greyMatter)

    #meanGreyMatterAntes = np.mean(np.mean(greyMaskLowDose, axis=-1), axis=-1)
    meanGreyMatterAntes = meanPerSlice(greyMaskGroundTruth)

    crcGroundTruthImagePerSlice = crcValuePerSlice(greyMaskGroundTruth[:, 0, :, :], greyMatter[:, 0, :, :],whiteMatter[:, 0, :, :])
    covGroundTruthImagePerSlice = covValuePerSlice(greyMaskGroundTruth[:, 0, :, :], greyMatter[:, 0, :, :])
    mseGroundTruthImagePerSlice = mseValuePerSlice(greyMaskGroundTruth[:, 0, :, :], groundTruthDataSet[:, 0, :, :],greyMatter[:, 0, :, :])

# Calculate input image metrics
if calculateInputImages:
    greyMaskLowDose = (noisyDataSetArray * greyMatter)
    greyMaskFullDose = (groundTruthDataSet * greyMatter)

    #meanGreyMatterAntes = np.mean(np.mean(greyMaskLowDose, axis=-1), axis=-1)
    meanGreyMatterAntes = meanPerSlice(greyMaskLowDose)
    stdGreyMatterAntes = np.std(np.std(greyMaskFullDose, axis=-1), axis=-1)

    crcInputImagePerSlice = crcValuePerSlice(noisyDataSetArray[:, 0, :, :], greyMatter[:, 0, :, :],whiteMatter[:, 0, :, :])
    covInputImagePerSlice = covValuePerSlice(noisyDataSetArray[:, 0, :, :], greyMatter[:, 0, :, :])
    mseInputImagePerSlice = mseValuePerSlice(noisyDataSetArray[:, 0, :, :], groundTruthDataSet[:, 0, :, :],greyMatter[:, 0, :, :])


# Calculate input images + filter metrics
if calculateFiltersValues:
    filtersFWHM_mm = np.array([2,4,6,8,10,12])
    filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0] # we use an isometric filter in voxels (using x dimension as voxel size, not ideal)
    filtersStdDev_voxels = filtersFWHM_voxels/2.35
    meanGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0], noisyDataSetArray.shape[1]))
    meanWhiteMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0], noisyDataSetArray.shape[1]))
    covFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
    crcFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
    mseFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
    mseGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))


    filterSub = []


    filtersFWHM_mm = np.array([2,4,6,8,10,12])
    filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0]
    filtersStdDev_voxels = filtersFWHM_voxels/2.35

    meanGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
    stdGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
    mseGreyMatter_filter = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))


    for fil in range(0, len(filtersStdDev_voxels)):
        filtStdDev_voxels = filtersStdDev_voxels[fil]
        # Now doing a 2D filter to match the 2D processing of the UNET:
        appfilter = (skimage.filters.gaussian(noisyDataSetArray[:,0,:,:], sigma=(0,filtStdDev_voxels,filtStdDev_voxels))).squeeze()

        #meanGreyMatterFilterPerSlice [fil,:] = np.mean(np.mean((appfilter*greyMatter[:,0,:,:]), axis = -1), axis=-1)

        meanGreyMatterFilterPerSlice[fil, :] = meanPerSlice(appfilter*greyMatter[:,0,:,:])
        stdGreyMatterFilterPerSlice [fil,:] = np.std(np.std((appfilter*greyMatter[:,0,:,:]), axis = -1), axis=-1)

        mseGreyMatterFilterPerSlice[fil,:] = mseValuePerSlice(appfilter,groundTruthDataSet[:, 0, :, :], greyMatter[:,0,:,:])

        covFilterPerSlice [fil,:] = covValuePerSlice(appfilter, greyMatter[:,0,:,:]).reshape(1,-1)
        crcFilterPerSlice [fil,:] = crcValuePerSlice(appfilter, greyMatter[:,0,:,:], whiteMatter[:,0,:,:]).reshape(1,-1)

        image = sitk.GetImageFromArray(np.array(appfilter))
        image.SetSpacing(voxelSize_mm)
        nameImage = 'filter' + str(filtersFWHM_mm[fil]) + '_dose_' + str(lowDose_perc) + '_OutModel_' + '.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)


outModel = np.zeros((noisyDataSetArray.shape[0], 1,noisyDataSetArray.shape[2], noisyDataSetArray.shape[2]))


if normSliceMeanStd:
    meanSliceTrainNoisy = np.mean(np.mean(noisyDataSetArray, axis=-1), axis=-1)
    stdSliceTrainNoisy = np.std(np.std(noisyDataSetArray, axis=-1), axis=-1)

    subjectNoisyNorm = (noisyDataSetArray [:,:,:, :]- meanSliceTrainNoisy[:, :,None, None]) / stdSliceTrainNoisy[:, :,None, None]

if calculateModelsValues:
    if normMeanSlice:
        meanSliceTrainNoisy = np.mean(np.mean(noisyDataSetArray, axis=-1), axis=-1)
        subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, :, None, None]

    if normMaxSlice:
        maxSliceTrainNoisy = np.max(np.max(noisyDataSetArray, axis=-1), axis=-1)
        subjectNoisyNorm = noisyDataSetArray [:,:,:, :]/ maxSliceTrainNoisy[:, :,None, None]

    if normBrainMean:
        BrainMask = whiteMatter + greyMatter
        BrainMask = np.where(BrainMask != 0, 1, 0)

        meanSliceTrainNoisy = (meanPerSlice(noisyDataSetArray[:, 0, :, :] * BrainMask[:, 0, :, :])).astype(np.float32)
        subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, None, None, None]

    if normMeanGreyMatter:
        BrainMask = np.where(greyMatter != 0, 1, 0)

        meanSliceTrainNoisy = (meanPerSlice(noisyDataSetArray[:, 0, :, :] * BrainMask[:, 0, :, :])).astype(np.float32)
        subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, None, None, None]

    if randonGaussian:
        datos = np.load(readNormData + 'Datos normalizacion.csv.npy')
        subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[None, None,:, :]

    if randomScale:
        datos = np.transpose(np.load(readNormData + 'Datos normalizacion.npy'))
        subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[:,:,None, None]


    if normMeanSlice == False and normSliceMeanStd == False and normMaxSlice == False and normBrainMean == False  and normMeanGreyMatter == False:
        subjectNoisyNorm = noisyDataSet

    ### Model
    for modelFilename in modelFilenames:
        model.load_state_dict(torch.load(modelsPath + modelFilename, map_location=torch.device(device)))

        if batchSubjects:
            # Divido el dataSet
            numBatches = np.round(subjectNoisyNorm.shape[0] / batchSubjectsSize).astype(int)
            # Run the model for all the slices:
            for i in range(numBatches):
                outModel[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :, :, :] = RunModel(model, torch.from_numpy(subjectNoisyNorm[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :, :, :])).detach().numpy()
            ndaOutputModel = outModel
        else:
            outModel = RunModel(model, torch.from_numpy(subjectNoisyNorm))
            # Convert it into numpy:
            ndaOutputModel = outModel.detach().numpy()

            ndaOutputModel = ndaOutputModel.squeeze()  # Remove the channel dimension
            # Unnormalize if using normalization:

        if normSliceMeanStd:
            ndaOutputModel = ndaOutputModel  * stdSliceTrainNoisy[:,:, None, None] + meanSliceTrainNoisy[:,:, None, None]

        if normMeanSlice:
            ndaOutputModel = ndaOutputModel * meanSliceTrainNoisy[:, :, None, None]

        if normBrainMean:
            ndaOutputModel = ndaOutputModel * meanSliceTrainNoisy[:, None, None, None]

        if normMaxSlice:
            ndaOutputModel = ndaOutputModel * maxSliceTrainNoisy[:, :, None, None]

        if randonGaussian:
            ndaOutputModel = ndaOutputModel / datos[None, None, :,:]

        if randomScale :
            ndaOutputModel = ndaOutputModel / datos[:,:,None, None]

    image = sitk.GetImageFromArray(np.array(ndaOutputModel.squeeze()))
    image.SetSpacing(voxelSize_mm)
    nameImage = 'Subject_dose_' + str(lowDose_perc) + '_OutModel_'  + '.nii'
    save_path = os.path.join(pathSaveResults, nameImage)
    sitk.WriteImage(image, save_path)

    greyMaskModelOutput = ((ndaOutputModel.squeeze()) * greyMatter[:,0,:,:])

    #meanGreyMatterDespues = np.mean(np.mean(greyMaskModelOutput, axis = -1), axis=-1)
    meanGreyMatterDespues =  meanPerSlice(greyMaskModelOutput)
    stdGreyMatterDespues = np.std(np.std(greyMaskModelOutput, axis = -1), axis=-1)

    #mseGreyMatterDespues = mseValuePerSlice(greyMaskFullDose,greyMaskModelOutput)

    crcInputImagePerSliceDespues = crcValuePerSlice(ndaOutputModel.squeeze(), greyMatter[:,0,:,:], whiteMatter[:,0,:,:])
    covInputImagePerSliceDespues = covValuePerSlice(ndaOutputModel.squeeze(), greyMatter[:,0,:,:])
    mseInputImagePerSliceDespues = mseValuePerSlice(ndaOutputModel.squeeze(), groundTruthDataSet[:,0,:,:], greyMatter[:,0,:,:])

filtersFWHM_mm = np.array([2,4,6,8])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0]
filtersStdDev_voxels = filtersFWHM_voxels/2.35


if saveDataCSV == True:

    if calculateInputImages:
        saveDataCsv(meanGreyMatterAntes ,'meanGreyMatterAntes' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(stdGreyMatterAntes,'stdGreyMatterAnte' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
        saveDataCsv(covInputImagePerSlice,'covInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(mseInputImagePerSlice,'mseInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

        saveDataCsv(crcInputImagePerSlice,'crcInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    if calculateModelsValues:
        saveDataCsv(meanGreyMatterDespues ,'meanGreyMatterDespues' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(stdGreyMatterDespues,'stdGreyMatterDespues' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
        saveDataCsv(mseInputImagePerSliceDespues, 'mseGreyMatterDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(covInputImagePerSliceDespues, 'covDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(crcInputImagePerSliceDespues, 'crcDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    if  calculateFiltersValues:
        saveDataCsv(meanGreyMatterFilterPerSlice.T,'meanGreyMatterFilter '+ ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(stdGreyMatterFilterPerSlice.T, 'stdInputImagePerSliceFilter' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(covFilterPerSlice.T, 'covPerSliceFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(crcFilterPerSlice.T, 'crcPerSliceFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
        saveDataCsv(mseGreyMatterFilterPerSlice.T, 'msePerSliceFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    if  calculateGroundTruthValues:
        saveDataCsv(meanGreyMatterAntes,'meanGreyMatterGroundTruth.csv',pathSaveResults)
        saveDataCsv(covGroundTruthImagePerSlice, 'covPerSliceGroundTruth.csv',pathSaveResults)
        saveDataCsv(crcGroundTruthImagePerSlice, 'crcPerSliceGroundTruth.csv',pathSaveResults)
        saveDataCsv(mseGroundTruthImagePerSlice, 'msePerSliceGroundTruth.csv',pathSaveResults)


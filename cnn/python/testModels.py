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
#from unet import UnetDe1a16Hasta512
from unet import UnetWithResidual
#from unet import UnetWithResidual5Layers
from utils import reshapeDataSet
from utils import mseValuePerSlice
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
batchSubjectsSize = 20
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
normMeanSlice = False
normMaxSlice = False
randonGaussian = True
randomScale = False

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################

######### PATHS ############


path = os.getcwd()

# model
nameModel = 'UnetResidual_MSE_lr5e-05_AlignTrue_randomScaleGaussian'.format(learning_rate)

modelsPath = '../../results/' + nameModel + '/models/'
readNormData = '../../results/' + nameModel +'/'

groundTruthSubdir = '100'
lowDoseSubdir = str(lowDose_perc)

# Output
pathSaveResults = '../../results/' + nameModel + '/'

# Data path
dataPath = '../../data/BrainWebSimulationsCompleteDataSet/'
pathPhantoms = dataPath + '/Phantoms/'


########### CREATE MODEL ###########
model =UnetWithResidual(1,1)

#model = Unet(1,1)


########## LIST OF MODELS ###############
modelFilenames = os.listdir(modelsPath)

########## PROCESSS DATA ###############

groundTruth = sitk.ReadImage(dataPath +'/validGroundTruth.nii')
voxelSize_mm = groundTruth.GetSpacing()
groundTruth = sitk.GetArrayFromImage(groundTruth)

noisyDataSet = sitk.ReadImage(dataPath +'/ValidNoisyDataset.nii')
noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

whiteMatter = sitk.ReadImage(dataPath +'/validWhiteMatterMask.nii')
whiteMatter = sitk.GetArrayFromImage(whiteMatter)

greyMatter  = sitk.ReadImage(dataPath +'/validGreyMatterMask.nii')
greyMatter  = sitk.GetArrayFromImage(greyMatter )

groundTruth = reshapeDataSet(groundTruth.squeeze())
noisyDataSet = reshapeDataSet(noisyDataSet.squeeze())
whiteMatter = reshapeDataSet(whiteMatter.squeeze())
greyMatter = reshapeDataSet(greyMatter.squeeze())


###

groundTruthDataSet = np.array(groundTruth)
noisyDataSetArray = np.array(noisyDataSet)
greyMatter = np.array(greyMatter)
whiteMatter = np.array(whiteMatter)

#noisyDataSet = np.expand_dims(noisyDataSetArray, axis=1)

# Input images + filter
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

greyMaskLowDose = (noisyDataSetArray * greyMatter)
greyMaskFullDose = (groundTruthDataSet * greyMatter)

meanGreyMatterAntes = np.mean(np.mean(greyMaskLowDose, axis = -1), axis=-1)
stdGreyMatterAntes = np.std(np.std(greyMaskFullDose, axis = -1), axis=-1)


crcInputImagePerSlice = crcValuePerSlice(noisyDataSetArray[:,0,:,:], greyMatter[:,0,:,:], whiteMatter[:,0,:,:])
covInputImagePerSlice = covValuePerSlice(noisyDataSetArray[:,0,:,:], greyMatter[:,0,:,:])
mseInputImagePerSlice = mseValuePerSlice(noisyDataSetArray[:,0,:,:], groundTruthDataSet[:,0,:,:], greyMatter[:,0,:,:])

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

    meanGreyMatterFilterPerSlice [fil,:] = np.mean(np.mean((appfilter*greyMatter[:,0,:,:]), axis = -1), axis=-1)
    stdGreyMatterFilterPerSlice [fil,:] = np.std(np.std((appfilter*greyMatter[:,0,:,:]), axis = -1), axis=-1)

    #mseGreyMatter_filter = mseValuePerSlice(appfilter*greyMatter[:,0,:,:],np.array(greyMaskFullDose), np.array([]))

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

if normMeanSlice:
    meanSliceTrainNoisy = np.mean(np.mean(noisyDataSetArray, axis=-1), axis=-1)
    subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, :, None, None]

if normMaxSlice:
    maxSliceTrainNoisy = np.max(np.max(noisyDataSetArray, axis=-1), axis=-1)
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]/ maxSliceTrainNoisy[:, :,None, None]

if randonGaussian:
    datos = np.load(readNormData + 'Datos normalizacion.csv.npy')
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[None, None,:, :]

if randomScale:
    datos = np.transpose(np.load(readNormData + 'Datos normalizacion.npy'))
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[:,:,None, None]


if normMeanSlice == False and normSliceMeanStd == False and normMaxSlice == False:
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

meanGreyMatterDespues = np.mean(np.mean(greyMaskModelOutput, axis = -1), axis=-1)
stdGreyMatterDespues = np.std(np.std(greyMaskModelOutput, axis = -1), axis=-1)

#mseGreyMatterDespues = mseValuePerSlice(greyMaskFullDose,greyMaskModelOutput)

crcInputImagePerSliceDespues = crcValuePerSlice(ndaOutputModel.squeeze(), greyMatter[:,0,:,:], whiteMatter[:,0,:,:])
covInputImagePerSliceDespues = covValuePerSlice(ndaOutputModel.squeeze(), greyMatter[:,0,:,:])
mseInputImagePerSliceDespues = mseValuePerSlice(ndaOutputModel.squeeze(), groundTruthDataSet[:,0,:,:], greyMatter[:,0,:,:])

filtersFWHM_mm = np.array([2,4,6,8])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0]
filtersStdDev_voxels = filtersFWHM_voxels/2.35


if saveDataCSV == True:

    saveDataCsv(meanGreyMatterAntes ,'meanGreyMatterAntes' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterAntes,'stdGreyMatterAnte' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
    saveDataCsv(covInputImagePerSlice,'covInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(mseInputImagePerSlice,'mseInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    saveDataCsv(crcInputImagePerSlice,'crcInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)


    saveDataCsv(meanGreyMatterDespues ,'meanGreyMatterDespues' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterDespues,'stdGreyMatterDespues' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
    #saveDataCsv(mseGreyMatterDespues, 'mseGreyMatterAntes ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(covInputImagePerSliceDespues, 'covDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(crcInputImagePerSliceDespues, 'crcDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    saveDataCsv(meanGreyMatterFilterPerSlice.T,'meanGreyMatterFilter '+ ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterFilterPerSlice.T, 'stdInputImagePerSliceFilter' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(mseFilterPerSlice.T,'mseInputImagePerSliceFilter' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(covFilterPerSlice.T, 'covFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(crcFilterPerSlice.T, 'crcFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
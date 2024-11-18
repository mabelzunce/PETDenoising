import torch
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import os
import SimpleITK as sitk
import numpy as np
import math

from cnn.python.utils import mseValuePerSubject
from unet3d import Unet
from unet3d import UnetWithResidual
from utils import crcValuePerSubject
from utils import covValuePerSubject
from utils import meanPerSubject
from utils import stdPerSubject
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

calculateOnlyValidSubjectMetrics = False
calculateOnlyTrainingSubjectMetrics = False
calculateAllSubjectMetrics = False
calculateFullDoseImage = False

# Subject

batchSubjects = False
batchSubjectsSize = 1
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

normMeanStd = False
normMean = False
normMeanBrain = True
normMax = False
randonGaussian = False
randomScale = False

######################### CHECK DEVICE ######################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################

######### PATHS ############
path = os.getcwd()

# model
nameModel = 'NUEVA_Unet_MSE_lr1e-05_AlignTrue_MeanBrain'.format(learning_rate)

modelsPath = '../../results/newContrast/3dRESULTS/' + nameModel + '/models/'
readNormData = '../../results/3dRESULTS/' + nameModel +'/'

groundTruthSubdir = '100'
lowDoseSubdir = str(lowDose_perc)

# Output
pathSaveResults = '../../results/newContrast/3dRESULTS/' + nameModel +'/'

# Data path
dataPath = '../../data/BrainWebSimulations3D/'


########### CREATE MODEL ###########
#model = UnetWithResidual(1,1)
model = Unet(1,1)


########## LIST OF MODELS ###############
modelFilenames = os.listdir(modelsPath)

########## PROCESSS DATA ###############

groundTruth = sitk.ReadImage(dataPath +'/testGroundTruth.nii')
#groundTruth = sitk.ReadImage(dataPath +'/testFullDose.nii')
voxelSize_mm = groundTruth.GetSpacing()
groundTruth = sitk.GetArrayFromImage(groundTruth)

noisyDataSet = sitk.ReadImage(dataPath +'/testNoisyDataset.nii')
#noisyDataSet = sitk.ReadImage(dataPath +'/testFullDose.nii')
noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)

whiteMatter = sitk.ReadImage(dataPath +'/testWhiteMatterMask.nii')
whiteMatter = sitk.GetArrayFromImage(whiteMatter)

greyMatter  = sitk.ReadImage(dataPath +'/testGreyMatterMask.nii')
greyMatter  = sitk.GetArrayFromImage(greyMatter )
###

# outputPath = '../../data/BrainWebSimulations3DDataset'
# arrayGroundTruth = []
# arrayNoisyDataSet = []
# arrayGreyMatterMask = []
# arrayWhiteMatterMask = []
# for subject in range(len(groundTruth)):
#     for slice in range(len(groundTruth[subject])):
#         arrayGroundTruth.append(groundTruth[subject,slice,:,:])
#         arrayNoisyDataSet.append(noisyDataSet[subject,slice,:,:])
#         arrayGreyMatterMask.append(greyMatter[subject,slice,:,:])
#         arrayWhiteMatterMask.append(whiteMatter[subject,slice,:,:])
#
# image = sitk.GetImageFromArray(np.array(arrayGroundTruth))
# image.SetSpacing(voxelSize_mm)
# nameImage = 'arrayGroundTruthPrueba.nii'
# save_path = os.path.join(outputPath, nameImage)
# sitk.WriteImage(image, save_path)
#
# image = sitk.GetImageFromArray(np.array(arrayNoisyDataSet))
# image.SetSpacing(voxelSize_mm)
# nameImage = 'arrayNoisyDataSetPrueba.nii'
# save_path = os.path.join(outputPath, nameImage)
# sitk.WriteImage(image, save_path)
#
# image = sitk.GetImageFromArray(np.array(arrayGreyMatterMask))
# image.SetSpacing(voxelSize_mm)
# nameImage = 'arrayGreyMatterMaskPrueba.nii'
# save_path = os.path.join(outputPath, nameImage)
# sitk.WriteImage(image, save_path)


# groundTruthDataSet = np.array(groundTruth[0:42,:,:,:])
# noisyDataSetArray = np.array(noisyDataSet[0:42,:,:,:])
# greyMatter = np.array(greyMatter[0:42,:,:,:])
# whiteMatter = np.array(whiteMatter[0:42,:,:,:])

groundTruthDataSet = (np.array(groundTruth)).astype(np.float32)
noisyDataSetArray = (np.array(noisyDataSet)).astype(np.float32)
greyMatter = (np.array(greyMatter)).astype(np.float32)
whiteMatter = (np.array(whiteMatter)).astype(np.float32)

greyMatter = np.where(greyMatter != 0, 1, 0)
whiteMatter = np.where(whiteMatter != 0, 1, 0)

#noisyDataSet = np.expand_dims(noisyDataSetArray, axis=1)

# Input image
covInputImagePerSubject = covValuePerSubject(noisyDataSetArray, greyMatter)
crcInputImagePerSubject = crcValuePerSubject(noisyDataSetArray, greyMatter, whiteMatter)
mseInputImagePerSlice = mseValuePerSubject(noisyDataSetArray, groundTruthDataSet, greyMatter)

subjectSave = [np.argmin(covInputImagePerSubject),np.argmax(covInputImagePerSubject)]
for i in subjectSave:
    image = sitk.GetImageFromArray(np.array((noisyDataSetArray[i])))
    image.SetSpacing(voxelSize_mm)
    nameImage = 'Input_dose_' + str(lowDose_perc) +'_index'+ str(i)  + '.nii'
    save_path = os.path.join(pathSaveResults, nameImage)
    sitk.WriteImage(image, save_path)

for i in subjectSave:
    image = sitk.GetImageFromArray(np.array((groundTruthDataSet)[i]))
    image.SetSpacing(voxelSize_mm)
    nameImage = 'GroundTruth_dose_' + str(lowDose_perc) +'_index'+ str(i)  + '.nii'
    save_path = os.path.join(pathSaveResults, nameImage)
    sitk.WriteImage(image, save_path)

# Input images + filter
filtersFWHM_mm = np.array([2,4,6,8,10,12])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0] # we use an isometric filter in voxels (using x dimension as voxel size, not ideal)
filtersStdDev_voxels = filtersFWHM_voxels/2.35
meanGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
stdGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
meanWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
covFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
crcFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
mseFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))
mseGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyDataSetArray.shape[0]))


filterSub = []
greyMaskLowDose = (noisyDataSetArray * greyMatter)
greyMaskFullDose = (groundTruthDataSet * greyMatter)

meanGreyMatterAntes = meanPerSubject(greyMaskLowDose)
#meanGreyMatterAntes = np.mean(np.mean(np.mean(greyMaskLowDose, axis = -1), axis=-1),axis = -1)
stdGreyMatterAntes = np.mean(np.std(np.std(greyMaskFullDose, axis = -1), axis=-1),axis=-1)
stdGreyMatterAntes = np.mean(np.std(np.std(greyMaskFullDose, axis = -1), axis=-1),axis=-1)

if calculateFullDoseImage == True:
    meanGreyMatterFullDose = meanPerSubject(greyMaskFullDose)
    covFullDose = covValuePerSubject(groundTruthDataSet, greyMatter)
    crcFullDose = crcValuePerSubject(groundTruthDataSet, greyMatter, whiteMatter)
    mseFullDose = mseValuePerSubject(groundTruthDataSet, noisyDataSetArray, greyMatter)


filtersFWHM_mm = np.array([2,4,6,8,10,12])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0]
filtersStdDev_voxels = filtersFWHM_voxels/2.35


for fil in range(0, len(filtersStdDev_voxels)):
    filtStdDev_voxels = filtersStdDev_voxels[fil]
    # Now doing a 2D filter to match the 2D processing of the UNET:
    appfilter = (skimage.filters.gaussian(noisyDataSetArray, sigma=(0,filtStdDev_voxels,filtStdDev_voxels,filtStdDev_voxels)))

    #meanGreyMatterFilterPerSubject [fil,:] = np.mean(np.mean(np.mean((appfilter*greyMatter), axis = -1), axis=-1), axis=-1)
    meanGreyMatterFilterPerSubject [fil,:] = meanPerSubject(appfilter*greyMatter)

    stdGreyMatterFilterPerSubject [fil,:] = np.std(np.std(np.std((appfilter*greyMatter), axis = -1), axis=-1), axis=-1)

    mseFilterPerSubject [fil,:]= mseValuePerSubject(appfilter,groundTruthDataSet,greyMatter)
    covFilterPerSubject [fil,:] = covValuePerSubject(appfilter, greyMatter)
    crcFilterPerSubject [fil,:] = crcValuePerSubject(appfilter, greyMatter, whiteMatter)

    for i in subjectSave:
        image = sitk.GetImageFromArray(np.array(appfilter[i]))
        image.SetSpacing(voxelSize_mm)
        nameImage = 'filter' + str(filtersFWHM_mm[fil]) + '_dose_' + str(lowDose_perc) + '_index'+ str(i)  + '.nii'
        save_path = os.path.join(pathSaveResults, nameImage)
        sitk.WriteImage(image, save_path)


outModel = np.zeros((noisyDataSetArray.shape[0], 1,noisyDataSetArray.shape[1], noisyDataSetArray.shape[2], noisyDataSetArray.shape[3]))


if normMeanStd:
    meanSliceTrainNoisy = np.mean(noisyDataSetArray,axis=(1, 2,3))
    stdSliceTrainNoisy = np.std(noisyDataSetArray,axis=(1, 2,3))

    subjectNoisyNorm = (noisyDataSetArray [:,:,:, :]- meanSliceTrainNoisy[:, None,None, None]) / stdSliceTrainNoisy[:, None,None, None]

if normMean:
    meanSliceTrainNoisy = np.mean(noisyDataSetArray, axis=(1, 2,3))
    subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, None, None, None]

if normMeanBrain:
    BrainMask = greyMatter + whiteMatter
    meanSliceTrainNoisy = meanPerSubject((noisyDataSetArray* BrainMask))
    subjectNoisyNorm = noisyDataSetArray[:, :, :, :] / meanSliceTrainNoisy[:, None, None, None]
    subjectNoisyNorm = (np.array(subjectNoisyNorm)).astype(np.float32)


if normMax:
    maxSliceTrainNoisy = np.max(noisyDataSetArray, axis=(1, 2,3))
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]/ maxSliceTrainNoisy[:, None,None, None]

if randonGaussian:
    datos = np.load(readNormData + 'Datos normalizacion.csv.npy')
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[None, None,:, :]

if randomScale:
    datos = np.transpose(np.load(readNormData + 'Datos normalizacion.npy'))
    subjectNoisyNorm = noisyDataSetArray [:,:,:, :]*datos[:,:,None, None]


if normMean == False and normMeanStd == False and normMax == False and normMeanBrain == False:
    subjectNoisyNorm = noisyDataSetArray

### Model
for modelFilename in modelFilenames:
    model.load_state_dict(torch.load(modelsPath + modelFilename, map_location=torch.device(device)))

    if batchSubjects:
        # Divido el dataSet
        numBatches = np.round(subjectNoisyNorm.shape[0] / batchSubjectsSize).astype(int)
        # Run the model for all the slices:
        for i in range(numBatches):
            outModel[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :,:, :, :] = RunModel(model, torch.from_numpy(subjectNoisyNorm[i * batchSubjectsSize: (i + 1) * batchSubjectsSize, :, :, :]).unsqueeze(1)).detach().numpy()
        ndaOutputModel = outModel
    else:
        outModel = RunModel(model, torch.from_numpy(subjectNoisyNorm).unsqueeze(1)).detach().numpy()
        # Convert it into numpy:
        ndaOutputModel = outModel
        # Unnormalize if using normalization:

    if normMeanStd:
        ndaOutputModel = ndaOutputModel * stdSliceTrainNoisy[:,None,None, None, None] + meanSliceTrainNoisy[:,None, None,None, None]

    if normMean:
        ndaOutputModel = ndaOutputModel * meanSliceTrainNoisy[:, None, None, None,None]

    if normMax:
        ndaOutputModel = ndaOutputModel * maxSliceTrainNoisy[:, None, None, None,None]

    if normMeanBrain:
        ndaOutputModel = ndaOutputModel * meanSliceTrainNoisy[:, None, None, None,None]

    if randonGaussian:
        ndaOutputModel = ndaOutputModel / datos[None, None, :,:]

    if randomScale :
        ndaOutputModel = ndaOutputModel / datos[:,:,None, None]

for i in subjectSave:
    image = sitk.GetImageFromArray(np.array((ndaOutputModel.squeeze())[i]))
    image.SetSpacing(voxelSize_mm)
    nameImage = 'Subject_dose_' + str(lowDose_perc) +'_OutModel_index'+ str(i)  + '.nii'
    save_path = os.path.join(pathSaveResults, nameImage)
    sitk.WriteImage(image, save_path)
# image = sitk.GetImageFromArray(np.array(ndaOutputModel.squeeze()))
# image.SetSpacing(voxelSize_mm)
# nameImage = 'Subject_dose_' + str(lowDose_perc) + '_OutModel_'  + '.nii'
# save_path = os.path.join(pathSaveResults, nameImage)
# sitk.WriteImage(image, save_path)

greyMaskModelOutput = ((ndaOutputModel.squeeze()) * greyMatter)

#meanGreyMatterDespues = np.mean(np.mean(np.mean(greyMaskModelOutput, axis = -1), axis=-1), axis=-1)
meanGreyMatterDespues = meanPerSubject(greyMaskModelOutput)
stdGreyMatterDespues = np.std(np.std(np.std(greyMaskModelOutput, axis = -1), axis=-1), axis=-1)

mseGreyMatterDespues = mseValuePerSubject(ndaOutputModel.squeeze(),groundTruthDataSet,greyMatter)

crcInputImagePerSliceDespues = crcValuePerSubject(ndaOutputModel.squeeze(), greyMatter, whiteMatter)
covInputImagePerSliceDespues = covValuePerSubject(ndaOutputModel.squeeze(), greyMatter)
#mseInputImagePerSliceDespues = mseValuePerSlice(ndaOutputModel.squeeze(), groundTruthDataSet[:,0,:,:], greyMatter[:,0,:,:])

if saveDataCSV == True:

    saveDataCsv(meanGreyMatterAntes ,'meanGreyMatterAntes' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterAntes,'stdGreyMatterAnte' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
    saveDataCsv(covInputImagePerSubject,'covInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(mseInputImagePerSlice,'mseInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(crcInputImagePerSubject,'crcInputImagePerSliceAntes' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)


    saveDataCsv(meanGreyMatterDespues ,'meanGreyMatterDespues' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterDespues,'stdGreyMatterDespues' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv', pathSaveResults)
    saveDataCsv(mseGreyMatterDespues, 'mseGreyMatterDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(covInputImagePerSliceDespues, 'covDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(crcInputImagePerSliceDespues, 'crcDespues ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

    saveDataCsv(meanGreyMatterFilterPerSubject.T,'meanGreyMatterFilter '+ ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(stdGreyMatterFilterPerSubject.T, 'stdInputImagePerSliceFilter' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(mseFilterPerSubject.T,'mseInputImagePerSliceFilter' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(covFilterPerSubject.T, 'covFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    saveDataCsv(crcFilterPerSubject.T, 'crcFilter ' + nameModel + ' _dose_' + str(lowDose_perc) + '.csv', pathSaveResults)

    # saveDataCsv(meanGreyMatterFullDose ,'meanGreyMatterFullDose' + nameModel + '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    # saveDataCsv(covFullDose,'covInputImagePerSliceFullDose' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    # saveDataCsv(mseFullDose,'mseInputImagePerSliceFullDose' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)
    # saveDataCsv(crcFullDose,'crcInputImagePerSliceFullDose' + nameModel +  '_dose_' + str(lowDose_perc) + '.csv',pathSaveResults)

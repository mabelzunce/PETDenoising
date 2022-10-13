import torch
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#from unetM import Unet
from unet import Unet
#from unet import UnetDe1a16Hasta512
#from unet import UnetWithResidual
#from unet import UnetWithResidual5Layers
from utils import reshapeDataSet
from utils import mseValuePerSlice
from utils import mseValuePerSubject
from utils import showDataPlot
from utils import showPlotGlobalData
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

######## CONFIG ###########
normalizeInput = True
normalizeInputMeanGlobal = True
normalizeInputMeanSlice = False
normalizeInputMaxSlice = False
learning_rate=0.00005
lowDose_perc = 5
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

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################

######### PATHS ############
path = os.getcwd()

# model
#nameModel = 'ResidualUnet5LayersWithoutRelu_MSE_lr{0}_AlignTrue'.format(learning_rate)
#nameModel = 'Unet5LayersNewArchitecture_MSE_lr{0}_AlignTrue'.format(learning_rate)

#nameModel = 'Unet5LayersNewArchitectureHasta512_MSE_lr5e-05_AlignTrue'.format(learning_rate)
#nameModel = 'Unet6LayersNewArchitectureDe1a16Hasta512_MSE_lr5e-05_AlignTrue'.format(learning_rate)

#nameModel = 'Unet5Layers_MSE_lr5e-05_AlignTrue'.format(learning_rate)

#nameModel = 'Unet5LayersNewArchitecture_MSE_lr5e-05_AlignTrue'.format(learning_rate)

nameModel = 'Unet5LayersNewArchitecture_MSE_lr5e-05_AlignTrue'.format(learning_rate)


if normalizeInputMeanSlice:
    nameModel = nameModel + '_norm'
if normalizeInputMeanGlobal:
    nameModel = nameModel+ '_GlobalMeanNorm_normMeanValue'


modelsPath = '../../results/' + nameModel + '/models/'

# Data path
dataPath = '../../data/BrainWebSimulations/'

groundTruthSubdir = '100'
lowDoseSubdir = str(lowDose_perc)

# Output
pathSaveResults = '../../results/' + nameModel + '/'

########### CREATE MODEL ###########
#model = Unet()
#model = UnetWithResidual(1,1)
#model = Unet512(1,1,32)
#model = UnetDe1a16Hasta512(1,1,16)
model = Unet(1,1)
#model = UnetWithResidual5Layers(1, 1)

########## LIST OF MODELS ###############
modelFilenames = os.listdir(modelsPath)

########## PROCESSS DATA ###############
# Data:
pathGroundTruth = dataPath + '/100'
arrayGroundTruth = os.listdir(pathGroundTruth)

pathNoisyDataSet = dataPath + str(lowDose_perc)
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)

pathPhantoms = dataPath + '/Phantoms/'

nameGroundTruth=[]
groundTruthArray = []
noisyImagesArray = []
greyMaskArray = []
whiteMaskArray = []

# Conjunto
if calculateOnlyValidSubjectMetrics:
    idxConjunto = validSubjects
    conjuntoAnalisis = 'ValidSubjects'
    analisisSub = 2

if calculateOnlyTrainingSubjectMetrics:
    idxConjunto = trainingSubjects
    conjuntoAnalisis = 'TrainingSubjects'
    analisisSub = 5

if calculateAllSubjectMetrics:
    idxConjunto = None
    conjuntoAnalisis = 'AllSubjects'
    analisisSub = 5

analisisSlice = 80

# leo los dataSet

meanGlobalSubjectNoisy = []
subjectsNames = []
noisyImagesArrayOrig = []

for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element

    # read GroundTruth
    groundTruthImg = sitk.ReadImage(pathGroundTruthElement)
    voxelSize_mm = groundTruthImg.GetSpacing()
    groundTruth = sitk.GetArrayFromImage(groundTruthImg)
    groundTruth = reshapeDataSet(groundTruth)

    name, extension = os.path.splitext(element)
    if extension == '.gz':
        name, extension2 = os.path.splitext(name)
        extension = extension2 + extension

    ind = name.find('Subject')
    name = name[ind + len('Subject'):]

    prueba = []

    if int(name) in idxConjunto:
        print(name)
        # read noisyDataSet
        nametrainNoisyDataSet = 'noisyDataSet' + str(lowDose_perc) + '_Subject' + name + '.nii'
        pathNoisyDataSetElement = pathNoisyDataSet + '/' + nametrainNoisyDataSet
        noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
        noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
        noisyDataSet = reshapeDataSet(noisyDataSet)


        # read greyMask
        nameGreyMask = 'Phantom_' + name + '_grey_matter.nii'
        #nameGreyMask = 'Subject' + name + 'GreyMask.nii'
        pathGreyMaskElement = pathPhantoms + '/' + nameGreyMask
        greyMask = sitk.ReadImage(pathGreyMaskElement)
        greyMask = sitk.GetArrayFromImage(greyMask)
        greyMask = reshapeDataSet(greyMask)

        # read whiteMask
        nameWhiteMask = 'Phantom_' + name + '_white_matter.nii'
        #nameWhiteMask = 'Subject' + name + 'WhiteMask.nii'
        pathWhiteMaskElement = pathPhantoms + '/' + nameWhiteMask
        whiteMask = sitk.ReadImage(pathWhiteMaskElement)
        whiteMask = sitk.GetArrayFromImage(whiteMask)
        whiteMask = reshapeDataSet(whiteMask)

        if normalizeInputMeanGlobal:
            subjectsNames.append(name)

            X = np.ma.masked_equal(noisyDataSet, 0)
            nonZeros = np.sum((~(X.mask)))
            pixelNonZeros = np.sum(noisyDataSet)
            meanGlobalSubjectNoisy.append(pixelNonZeros / nonZeros)

            noisyDataSet = noisyDataSet / meanGlobalSubjectNoisy[-1]

            noisyImagesArrayOrig.append(noisyDataSet)

        nameGroundTruth.append(name)
        groundTruthArray.append(groundTruth)
        noisyImagesArray.append(noisyDataSet)
        greyMaskArray.append(greyMask)
        whiteMaskArray.append(whiteMask)

noisyImagesArray = np.array(noisyImagesArray)
groundTruthArray = np.array(groundTruthArray)
whiteMaskArray = np.array(whiteMaskArray)
greyMaskArray = np.array(greyMaskArray)

if normalizeInputMeanGlobal:
    noisyImagesArrayOrig = np.array(noisyImagesArrayOrig)

# Get the maximum value per slice:
# Normalize the input if necessary:
if normalizeInputMeanSlice:
    noisyImagesArrayOrig = noisyImagesArray

    meanSlice = noisyImagesArray[:, :, :, :].mean(axis=4).mean(axis=3)
    meanSliceGroundTruth = groundTruthArray[:, :, :, :].mean(axis=4).mean(axis=3)
    noisyImagesArray = noisyImagesArray / meanSlice
    noisyImagesArray = np.nan_to_num(noisyImagesArray)

    noisyImagesArrayOrig = noisyImagesArray

if normalizeInputMaxSlice:
    noisyImagesArrayOrig = noisyImagesArray

    maxSlice = noisyImagesArray[:, :, :, :].max(axis=4).max(axis=3)
    maxSliceGroundTruth = groundTruthArray[:, :, :, :].max(axis=4).max(axis=3)
    noisyImagesArray = noisyImagesArray / maxSlice
    noisyImagesArray = np.nan_to_num(noisyImagesArray)



contModel = 0
modelName = []


# Input images
meanGreyMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanWhiteMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanGreyMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
meanWhiteMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
meanGreyMatterInputImageGlobal = np.zeros((1))
meanWhiteMatterInputImageGlobal = np.zeros((1))
covInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
covInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
stdGreyMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
stdWhiteMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
mseInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
mseGreyMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseWhiteMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseGreyMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
mseWhiteMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))

crcInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
crcInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
covInputImageGlobal = np.zeros((1))
crcInputImageGlobal = np.zeros((1))
mseInputImageGlobal = np.zeros((1))
mseGreyMatterInputImageGlobal = np.zeros((1))
stdGreyMatterInputImageGlobal = np.zeros((1))
stdWhiteMatterInputImageGlobal = np.zeros((1))

# Input images + filter
filtersFWHM_mm = np.array([2,4,6,8])
filtersFWHM_voxels = filtersFWHM_mm/voxelSize_mm[0] # we use an isometric filter in voxels (using x dimension as voxel size, not ideal)
filtersStdDev_voxels = filtersFWHM_voxels/2.35
meanGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanWhiteMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
meanWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
covFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
crcFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseGreyMatterFilterPerSlice = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
covFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
crcFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
mseFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
stdGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
stdWhiteMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
mseGreyMatterFilterPerSubject = np.zeros((len(filtersFWHM_mm),noisyImagesArray.shape[0]))
crcFilterGlobal = np.zeros((len(filtersFWHM_mm)))
covFilterGlobal = np.zeros((len(filtersFWHM_mm)))
mseFilterGlobal = np.zeros((len(filtersFWHM_mm)))
mseGreyMatterFilterGlobal = np.zeros((len(filtersFWHM_mm)))
meanGreyMatterFilterGlobal = np.zeros((len(filtersFWHM_mm)))
meanWhiteMatterFilterGlobal = np.zeros((len(filtersFWHM_mm)))
stdGreyMatterFilterGlobal = np.zeros((len(filtersFWHM_mm)))
stdWhiteMatterFilterGlobal = np.zeros((len(filtersFWHM_mm)))


filterSub = []

for sub in range(0, len(noisyImagesArrayOrig)):
    # Get images for one subject as a torch tensor:
    noisyImagesSubject = noisyImagesArrayOrig[sub, :, :, :, :].squeeze()
    groundTruthSubject = groundTruthArray[sub, :, :, :, :].squeeze() # Remove the channel dimension
    whiteMaskSubject = whiteMaskArray[sub, :, :, :, :].squeeze()
    greyMaskSubject = greyMaskArray[sub, :, :, :, :].squeeze()
    print('Subject ', idxConjunto[sub])
    # calculo los filtros
    # los voy a guardar

    # METRICAS INPUT IMAGE
    mask = (noisyImagesSubject * greyMaskSubject)
    meanGreyMatterInputImagePerSlice[sub, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    mseGreyMatterInputImagePerSlice[sub, :] = mseValuePerSlice(noisyImagesSubject, groundTruthSubject, greyMaskSubject)
    mseGreyMatterInputImagePerSubject[sub] = mseValuePerSubject(noisyImagesSubject, groundTruthSubject, greyMaskSubject)
    mask = (noisyImagesSubject * whiteMaskSubject)
    meanWhiteMatterInputImagePerSlice[sub, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    crcInputImagePerSlice[sub, :] = crcValuePerSlice(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSlice[sub, :] = covValuePerSlice(noisyImagesSubject, greyMaskSubject)
    mseInputImagePerSlice[sub, :] = mseValuePerSlice(noisyImagesSubject,groundTruthSubject, greyMaskSubject)
    crcInputImagePerSubject[sub] = crcValuePerSubject(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSubject[sub] = covValuePerSubject(noisyImagesSubject, greyMaskSubject)
    mseInputImagePerSubject[sub] = mseValuePerSubject(noisyImagesSubject, groundTruthSubject, greyMaskSubject)

    crcInputImageGlobal = np.mean(crcInputImagePerSubject[:])
    covInputImageGlobal = np.mean(covInputImagePerSubject[:])

    meanWhiteMatterInputImagePerSubject[sub] = meanPerSubject(meanWhiteMatterInputImagePerSlice[sub, :])
    meanGreyMatterInputImagePerSubject[sub] = meanPerSubject(meanGreyMatterInputImagePerSlice[sub, :])
    stdGreyMatterInputImagePerSubject[sub] = stdPerSubject(noisyImagesSubject * greyMaskSubject)
    stdWhiteMatterInputImagePerSubject[sub] = stdPerSubject(noisyImagesSubject * whiteMaskSubject)
    meanGreyMatterInputImageGlobal = np.mean(meanGreyMatterInputImagePerSubject[:])
    meanWhiteMatterInputImageGlobal = np.mean(meanWhiteMatterInputImagePerSubject[:])
    mseInputImageGlobal = np.mean(mseInputImagePerSubject[:]) # remove
    mseGreyMatterInputImageGlobal = np.mean(mseGreyMatterInputImagePerSubject[:])
    stdGreyMatterInputImageGlobal = np.mean(stdGreyMatterInputImagePerSubject[:])
    stdWhiteMatterInputImageGlobal = np.mean(stdWhiteMatterInputImagePerSubject[:])

    # METRICAS FILTROS + INPUT IMAGE
    for fil in range(0, len(filtersStdDev_voxels)):
        filtStdDev_voxels = filtersStdDev_voxels[fil]
        # Now doing a 2D filter to match the 2D processing of the UNET:
        filter = (skimage.filters.gaussian(noisyImagesSubject, sigma=(0,filtStdDev_voxels,filtStdDev_voxels))).squeeze()

        if saveFilterOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(filter))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Subject' + str(nameGroundTruth[sub]) +'_dose_'+str(lowDose_perc)+'_filter_'+str(fil)+'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        if saveFilterOutputAsNiftiImageOneSubject and (analisisSub == int(nameGroundTruth[sub])):
            image = sitk.GetImageFromArray(np.array(filter))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Subject' + str(nameGroundTruth[sub]) +'_dose_'+str(lowDose_perc)+'_filter_'+str(fil)+'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        if int(nameGroundTruth[sub]) == analisisSub:
            filterSub.append(filter[analisisSlice,:,:])

        mseFilterPerSlice[fil,sub,:]=mseValuePerSlice(filter,groundTruthSubject)

        mask = (filter * greyMaskSubject)
        meanGreyMatterFilterPerSlice[fil,sub,:] = meanPerSlice((mask.reshape(mask.shape[0], -1)))
        mseGreyMatterFilterPerSlice[fil,sub,:] = mseValuePerSlice(filter, groundTruthSubject, greyMaskSubject)
        mseGreyMatterFilterPerSubject[fil, sub] = mseValuePerSubject(filter, groundTruthSubject, greyMaskSubject)
        mask= (filter * whiteMaskSubject)
        meanWhiteMatterFilterPerSlice[fil,sub, :] = meanPerSlice((mask.reshape(mask.shape[0], -1)))

        crcFilterPerSlice[fil, sub, :] = crcValuePerSlice(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSlice[fil, sub, :] = covValuePerSlice(filter, greyMaskSubject)

        crcFilterPerSubject[fil, sub] = crcValuePerSubject(filter, greyMaskSubject, whiteMaskSubject)

        covFilterPerSubject[fil, sub] = covValuePerSubject(filter, greyMaskSubject)
        mseFilterPerSubject[fil, sub] = mseValuePerSubject(filter, groundTruthSubject)
        meanGreyMatterFilterPerSubject[fil, sub] = meanPerSubject(meanGreyMatterFilterPerSlice[fil, sub, :])
        meanWhiteMatterFilterPerSubject[fil, sub] = meanPerSubject(meanWhiteMatterFilterPerSlice[fil, sub, :])
        stdGreyMatterFilterPerSubject[fil, sub] = stdPerSubject(filter * greyMaskSubject)
        stdWhiteMatterFilterPerSubject[fil, sub] = stdPerSubject(filter * whiteMaskSubject)

        crcFilterGlobal[fil] = np.mean(crcFilterPerSubject[fil, :])
        covFilterGlobal[fil] = np.mean(covFilterPerSubject[fil, :])

        meanGreyMatterFilterGlobal[fil] = np.mean(meanGreyMatterFilterPerSubject[fil, :])
        meanWhiteMatterFilterGlobal[fil] = np.mean(meanWhiteMatterFilterPerSubject[fil, :])

        mseFilterGlobal[fil] = np.mean(mseFilterPerSubject[fil, :])
        mseGreyMatterFilterGlobal[fil] = np.mean(mseGreyMatterFilterPerSubject[fil, :])

        stdGreyMatterFilterGlobal[fil] = np.mean(stdGreyMatterFilterPerSubject[fil, :])
        stdWhiteMatterFilterGlobal[fil] = np.mean(stdWhiteMatterFilterPerSubject[fil, :])

# Input images + models
allModelsCrc = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsCov = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanGM = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanWM = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanGMperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsMeanWMperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsCOVperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsCRCperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsMsePerSlice = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMsePerSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsGreyMatterMsePerSlice = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsGreyMatterMsePerSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsStdGreyMatterPerSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsStdWhiteMatterPerSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))

# Resultados globales
allModelsCOVGlobal = np.zeros((len(modelFilenames), 1))
allModelsCRCGlobal = np.zeros((len(modelFilenames), 1))
allModelsMeanGMGlobal = np.zeros((len(modelFilenames), 1))
allModelsMeanWMGlobal = np.zeros((len(modelFilenames), 1))
allModelsMseGlobal = np.zeros((len(modelFilenames), 1))
allModelsGreyMatterMseGlobal = np.zeros((len(modelFilenames), 1))
allModelsStdGreyMatterGlobal = np.zeros((len(modelFilenames), 1))
allModelsStdWhiteMatterGlobal = np.zeros((len(modelFilenames), 1))

outModel = np.zeros((noisyImagesArray.shape[1], 1,noisyImagesArray.shape[3], noisyImagesArray.shape[3]))
for modelFilename in modelFilenames:
    model.load_state_dict(torch.load(modelsPath + modelFilename, map_location=torch.device(device)))
    modelName.append(modelFilename)

    print('Model',contModel+1)

    for sub in range(0, len(noisyImagesArray)):
        # Get images for one subject as a torch tensor:
        noisyImagesSubject = noisyImagesArray[sub, :, :, :, :]
        groundTruthSubject = groundTruthArray[sub, :, :, :, :].squeeze() # Remove the channel dimension
        whiteMaskSubject = whiteMaskArray[sub, :, :, :, :].squeeze()
        greyMaskSubject = greyMaskArray[sub, :, :, :, :].squeeze()

        if normalizeInputMaxSlice:
            normSubject = maxSlice[sub, :].squeeze()

        if normalizeInputMeanSlice:
            normSubject = meanSlice[sub, :].squeeze()

        if normalizeInputMeanGlobal:
            normSubject = meanGlobalSubjectNoisy[sub]

        print('Subject ', idxConjunto[sub])

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

        ndaOutputModel = ndaOutputModel.squeeze()  # Remove the channel dimension
        # Unnormalize if using normalization:
        if normalizeInputMeanSlice:
            #ndaOutputModel = ndaOutputModel * maxSliceSubject[:,None,None]
            ndaOutputModel = ndaOutputModel * normSubject[:, None, None]

        if normalizeInputMaxSlice:
            ndaOutputModel = ndaOutputModel * normSubject[:,None,None]

        if normalizeInputMeanGlobal:
            ndaOutputModel = ndaOutputModel * normSubject

        if saveModelOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(ndaOutputModel))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Subject' + str(nameGroundTruth[sub]) + '_dose_' + str(lowDose_perc) + '_OutModel_' + modelName[-1]  + '.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        if saveModelOutputAsNiftiImageOneSubject and (analisisSub == int(nameGroundTruth[sub])):
            image = sitk.GetImageFromArray(np.array(ndaOutputModel))
            image.SetSpacing(voxelSize_mm)
            nameImage = 'Subject' + str(nameGroundTruth[sub]) +'_dose_'+str(lowDose_perc)+'_OutModel_'+modelName[-1] +'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        if int(nameGroundTruth[sub]) == analisisSub:
            outputSub = ndaOutputModel[analisisSlice,:,:]
        # Compute metrics for each slice:
        greyMaskedImage = (ndaOutputModel * greyMaskSubject)
        whiteMaskedImage = (ndaOutputModel * whiteMaskSubject)

        # Compute metrics for each subject all slices:
        allModelsMeanGM[contModel, sub,  :] = meanPerSlice((whiteMaskedImage.reshape(whiteMaskedImage.shape[0], -1)))
        allModelsCrc[contModel, sub, :] = crcValuePerSlice(ndaOutputModel, greyMaskSubject, whiteMaskSubject)
        allModelsCov[contModel, sub,:] = covValuePerSlice(ndaOutputModel, greyMaskSubject)
        allModelsMsePerSlice[contModel,sub,:] = mseValuePerSlice(ndaOutputModel, groundTruthSubject)
        allModelsGreyMatterMsePerSlice[contModel, sub, :] = mseValuePerSlice(ndaOutputModel, groundTruthSubject, greyMaskSubject)

        # Compute metrics for all subject :
        allModelsCRCperSubject[contModel,sub] = crcValuePerSubject(ndaOutputModel,greyMaskSubject,whiteMaskSubject)
        allModelsCOVperSubject[contModel,sub] = covValuePerSubject(ndaOutputModel, greyMaskSubject)
        allModelsMeanGMperSubject[contModel, sub] = meanPerSubject(allModelsMeanGM[contModel, sub, :])
        allModelsMeanWMperSubject[contModel, sub] = meanPerSubject(allModelsMeanWM[contModel, sub, :])
        allModelsStdGreyMatterPerSubject[contModel, sub] = stdPerSubject(ndaOutputModel * greyMaskSubject)
        allModelsStdWhiteMatterPerSubject[contModel, sub] = stdPerSubject(ndaOutputModel * whiteMaskSubject)
        allModelsMsePerSubject[contModel, sub] = mseValuePerSubject(ndaOutputModel, groundTruthSubject)
        allModelsGreyMatterMsePerSubject[contModel, sub] = mseValuePerSubject(ndaOutputModel, groundTruthSubject, greyMaskSubject)

    # Resultados globales

    allModelsCOVGlobal[contModel] = np.mean(allModelsCOVperSubject[contModel,:])
    allModelsCRCGlobal[contModel] = np.mean(allModelsCRCperSubject[contModel,:])
    allModelsMeanGMGlobal[contModel] = np.mean(allModelsMeanGMperSubject[contModel,:])
    allModelsMeanWMGlobal[contModel] = np.mean(allModelsMeanWMperSubject[contModel,:])
    allModelsMseGlobal[contModel] = np.mean(allModelsMsePerSubject[contModel, :])
    allModelsGreyMatterMseGlobal[contModel] = np.mean(allModelsGreyMatterMsePerSubject[contModel, :])
    allModelsStdGreyMatterGlobal[contModel]= np.mean(allModelsStdGreyMatterPerSubject[contModel, :])
    allModelsStdWhiteMatterGlobal[contModel] = np.mean(allModelsStdGreyMatterPerSubject[contModel, :])

    contModel = contModel + 1

# Show plot
if showGlobalPlots == True:
    namesPlot = ['COV antes', 'COV modelos', 'COV filtros']
    showPlotGlobalData(covInputImageGlobal,allModelsCOVGlobal,covFilterGlobal, filtersFWHM_mm,
                 namesModel=modelName,graphName = 'Cov all models',names=namesPlot, saveFig=True,
                       pathSave=pathSaveResults)
    namesPlot = ['CRC antes', 'CRC modelos', 'CRC filtros']
    showPlotGlobalData(crcInputImageGlobal, allModelsCRCGlobal, crcFilterGlobal, filtersFWHM_mm,
                       namesModel=modelName, graphName='Crc all models', names=namesPlot, saveFig=True,
                       pathSave=pathSaveResults)
    namesPlot = ['MSE antes', 'MSE modelos', 'MSE filtros']
    showPlotGlobalData(mseGreyMatterInputImageGlobal, allModelsGreyMatterMseGlobal, mseGreyMatterFilterGlobal,
                 filtersFWHM_mm, namesModel=modelName, graphName='MSE Grey Matter all models',
                 names=namesPlot,saveFig = True, pathSave=pathSaveResults)
    namesPlot = ['MSE antes', 'MSE modelos', 'MSE filtros']
    showPlotGlobalData(mseInputImageGlobal, allModelsMseGlobal, mseFilterGlobal,
                       filtersFWHM_mm, namesModel=modelName, graphName='MSE all models',
                       names=namesPlot, saveFig=True, pathSave=pathSaveResults)

if showPerfilSlices == True:
    namesPlot = ['Mean antes', 'Mean model', 'Mean filtro']
    meanFilter = meanGreyMatterFilterPerSlice[:, :, :].mean(axis=2)
    meanOutModel = allModelsMeanGM[:, :, :].mean(axis=2)
    meanInputImage = meanGreyMatterInputImagePerSlice[:, :].mean(axis=1)
    showDataPlot(meanGreyMatterInputImagePerSlice[analisisSub, :] / meanInputImage[analisisSub],
                 allModelsMeanGM[:, analisisSub, :] / meanOutModel[:, analisisSub, None],
                 meanGreyMatterFilterPerSlice[:, analisisSub, :] / meanFilter[:, analisisSub, None]
                 , filtersFWHM_mm, graphName='Mean Grey Matter',
                 names=namesPlot,namesModel = modelName, saveFig = True, pathSave=pathSaveResults)

if showImageSub == True:
    noisySub = noisyImagesArray[analisisSub, analisisSlice, 0, :, :]
    gtSub = groundTruthArray[analisisSub, analisisSlice, 0, :, :]
    totalImg = [gtSub, noisySub]
    namesPlot = ['groundTruth', 'noisyImage']
    if len(outputSub) < (noisyImagesArray.shape[-1]):
        for i in range(0, len(outputSub)):
            totalImg.append(outputSub[i])
            namesPlot.append('Model')
    else:
        totalImg.append(outputSub)
        namesPlot.append('Model')
    if len(filterSub) < (noisyImagesArray.shape[-1]):
        for i in range(0, len(filterSub)):
            totalImg.append(filterSub[i])
            namesPlot.append('Filter')
    else:
        totalImg.append(filterSub)
        namesPlot.append('Filter')
    cantImg = len(totalImg)
    fig = plt.figure(figsize=(4, 4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, cantImg),
                     axes_pad=0.1,
                     )
    for ax, im in zip(grid, totalImg):
        img = ax.imshow(im, cmap='gray')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    plt.show(block=False)
    plt.savefig(pathSaveResults + 'ResultsModel')

if saveDataCSV == True:
    saveDataCsv(meanGreyMatterFilterGlobal, 'meanGreyMatterFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(meanGreyMatterInputImageGlobal, 'meanGreyMatterInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsMeanGMGlobal, 'meanGreyMatterAllModelsGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)

    saveDataCsv(covFilterGlobal, 'covFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(covInputImageGlobal, 'covInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsCOVGlobal, 'covAllModelsGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)

    saveDataCsv(stdGreyMatterFilterGlobal, 'stdGreyMatterFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(stdGreyMatterInputImageGlobal, 'stdGreyMatterInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsStdGreyMatterGlobal, 'stdGreyMatterAllModelsGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)

    saveDataCsv(stdWhiteMatterFilterGlobal, 'stdWhiteMatterFilterGlobal' + nameModel + conjuntoAnalisis +'_dose_'+str(lowDose_perc)+ '.csv', pathSaveResults)
    saveDataCsv(stdWhiteMatterInputImageGlobal, 'stdWhiteMatterInputImageGlobal' + nameModel + conjuntoAnalisis +'_dose_'+str(lowDose_perc)+ '.csv', pathSaveResults)
    saveDataCsv(allModelsStdWhiteMatterGlobal, 'stdWhiteMatterAllModelsGlobal' + nameModel + conjuntoAnalisis +'_dose_'+str(lowDose_perc)+ '.csv', pathSaveResults)

    saveDataCsv(crcFilterGlobal, 'crcFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(crcInputImageGlobal, 'crcInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsCRCGlobal, 'crcAllModelsGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)

    saveDataCsv(mseFilterGlobal, 'mseFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(mseInputImageGlobal, 'mseInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsMseGlobal, 'mseAllModelsGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)

    saveDataCsv(mseGreyMatterFilterGlobal, 'mseGreyMatterFilterGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(mseGreyMatterInputImageGlobal, 'mseGreyMatterInputImageGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
    saveDataCsv(allModelsGreyMatterMseGlobal, 'allModelsGreyMatterMseGlobal'+nameModel+conjuntoAnalisis+'_dose_'+str(lowDose_perc)+'.csv', pathSaveResults)
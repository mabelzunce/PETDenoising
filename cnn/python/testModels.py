import torch
import torchvision
import skimage
import matplotlib.pyplot as plt

#from unetM import Unet
#from unet import Unet
from unet import UnetWithResidual
from utils import imshow
from utils import reshapeDataSet
from utils import mseValuePerSlice
from utils import mseValuePerSubject
from utils import torchToNp
from utils import mseAntDspModelTorchSlice
from utils import testModelSlice
from utils import obtenerMask
from utils import showDataPlot
from utils import saveNumpyAsNii
from utils import getTestOneModelOneSlices
from utils import covValue
from utils import crcValue
from utils import covValuePerSlice
from utils import crcValuePerSlice
from utils import crcValuePerSubject
from utils import covValuePerSubject
from utils import meanPerSlice
from utils import meanPerSubject
from utils import RunModel
from utils import saveDataCsv
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

######## CONFIG ###########
normalizeInput = True
learning_rate=0.00005
lowDose_perc = 5
epoch = 23
actScaleFactor = 100/lowDose_perc
allSubjects = [*range(1,21)]
validSubjects = [2, 4, 6, 8]
trainingSubjects = allSubjects
for i in validSubjects:
    trainingSubjects.remove(i)

batchSubjects = True
batchSubjectsSize = 20
# Results visualization
showCovPlot = True
showPerfilSlices = True
# Save results
saveModelOutputAsNiftiImage = False
saveFilterOutputAsNiftiImage = False
saveCSV = True
###########################

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################

######### PATHS ############
path = os.getcwd()

# model
nameModel = 'UnetWithResidual_MSE_lr{0}_AlignTrue_norm'.format(learning_rate)
#nameModel = 'UnetWithResidual_MSE_lr{0}'.format(learning_rate)
#modelsPath = '../../../Results/' + nameModel + '/Models/'
modelsPath = 'C:/Users/Encargado/Desktop/Milagros/Results/' + nameModel + '/Models/'
modelFilename = modelsPath + 'UnetWithResidual_MSE_lr5e-05_AlignTrue_norm_20220715_191324_27_best_fit' #nameModel + str(epoch) + '_best_fit'

# Data path
#dataPath = 'D:/UNSAM/PET/BrainWebSimulations/'

dataPath = 'C:/Users/Encargado/Desktop/Milagros/NewDataset/'
groundTruthSubdir = '100'
lowDoseSubdir = str(lowDose_perc)

# Output
pathSaveResults = 'C:/Users/Encargado/Desktop/Milagros/Results/' + nameModel + '/'
#pathSaveResults = '../../../Results/' + nameModel + '/'

########### CREATE MODEL ###########
model = UnetWithResidual(1,1) # model = Unet(1,1)


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

# leo los dataSet
for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element

    # read GroundTruth
    groundTruthImg = sitk.ReadImage(pathGroundTruthElement)
    voxelSize = groundTruthImg.GetSpacing()
    groundTruth = sitk.GetArrayFromImage(groundTruthImg)
    groundTruth = reshapeDataSet(groundTruth)

    name, extension = os.path.splitext(element)
    if extension == '.gz':
        name, extension2 = os.path.splitext(name)
        extension = extension2 + extension

    ind = name.find('Subject')
    name = name[ind + len('Subject'):]

    # read noisyDataSet
    nametrainNoisyDataSet = 'noisyDataSet' + str(lowDose_perc) + '_Subject' + name + '.nii'
    pathNoisyDataSetElement = pathNoisyDataSet + '/' + nametrainNoisyDataSet
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    noisyDataSet = reshapeDataSet(noisyDataSet)

    # read greyMask
    #nameGreyMask = 'Phantom_' + name + '_grey_matter.nii'
    nameGreyMask = 'Subject' + name + 'GreyMask.nii'
    pathGreyMaskElement = pathPhantoms + '/' + nameGreyMask
    greyMask = sitk.ReadImage(pathGreyMaskElement)
    greyMask = sitk.GetArrayFromImage(greyMask)
    greyMask = reshapeDataSet(greyMask)

    # read whiteMask
    #nameWhiteMask = 'Phantom_' + name + '_white_matter.nii'
    nameWhiteMask = 'Subject' + name + 'WhiteMask.nii'
    pathWhiteMaskElement = pathPhantoms + '/' + nameWhiteMask
    whiteMask = sitk.ReadImage(pathWhiteMaskElement)
    whiteMask = sitk.GetArrayFromImage(whiteMask)
    whiteMask = reshapeDataSet(whiteMask)

    nameGroundTruth.append(name)
    groundTruthArray.append(groundTruth)
    noisyImagesArray.append(noisyDataSet)
    greyMaskArray.append(greyMask)
    whiteMaskArray.append(whiteMask)

noisyImagesArray = np.array(noisyImagesArray)
groundTruthArray = np.array(groundTruthArray)
whiteMaskArray = np.array(whiteMaskArray)
greyMaskArray = np.array(greyMaskArray)

noisyImagesArrayOrig = noisyImagesArray

# Get the maximum value per slice:
maxSlice = noisyImagesArray[:, :, :, :].max(axis=4).max(axis=3)
maxSliceGroundTruth = groundTruthArray[:, :, :, :].max(axis=4).max(axis=3)
# Normalize the input if necessary:
if normalizeInput:
    noisyImagesArray = noisyImagesArray/maxSlice[:,:,:,None,None]
    noisyImagesArray = np.nan_to_num(noisyImagesArray)
contModel = 0
modelName = []

## App filtros

# Input images
meanGreyMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanWhiteMatterInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanGreyMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
meanWhiteMatterInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
meanGreyMatterInputImageGlobal = np.zeros((1))
meanWhiteMatterInputImageGlobal = np.zeros((1))
covInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
covInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
mseInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
crcInputImagePerSlice = np.zeros((noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
crcInputImagePerSubject = np.zeros((noisyImagesArray.shape[0]))
covInputImageGlobal = np.zeros((1))
crcInputImageGlobal = np.zeros((1))
mseInputImageGlobal = np.zeros((1))

# Input images + filter
filters = [2,3,4]
meanGreyMatterFilterPerSlice = np.zeros((len(filters),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanWhiteMatterFilterPerSlice = np.zeros((len(filters),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
meanGreyMatterFilterPerSubject = np.zeros((len(filters),noisyImagesArray.shape[0]))
meanWhiteMatterFilterPerSubject = np.zeros((len(filters),noisyImagesArray.shape[0]))
covFilterPerSlice = np.zeros((len(filters),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
crcFilterPerSlice = np.zeros((len(filters),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
mseFilterPerSlice = np.zeros((len(filters),noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
covFilterPerSubject = np.zeros((len(filters),noisyImagesArray.shape[0]))
crcFilterPerSubject = np.zeros((len(filters),noisyImagesArray.shape[0]))
mseFilterPerSubject = np.zeros((len(filters),noisyImagesArray.shape[0]))
crcFilterGlobal = np.zeros((len(filters)))
covFilterGlobal = np.zeros((len(filters)))
mseFilterGlobal = np.zeros((len(filters)))
meanGreyMatterFilterGlobal = np.zeros((len(filters)))
meanWhiteMatterFilterGlobal = np.zeros((len(filters)))

for sub in range(0, len(noisyImagesArrayOrig)):
    # Get images for one subject as a torch tensor:
    noisyImagesSubject = noisyImagesArrayOrig[sub, :, :, :, :].squeeze()
    groundTruthSubject = groundTruthArray[sub, :, :, :, :].squeeze() # Remove the channel dimension
    whiteMaskSubject = whiteMaskArray[sub, :, :, :, :].squeeze()
    greyMaskSubject = greyMaskArray[sub, :, :, :, :].squeeze()
    print('Subject ', sub)
    # calculo los filtros
    # los voy a guardar

    # METRICAS INPUT IMAGE
    mask = (noisyImagesSubject * greyMaskSubject)
    meanGreyMatterInputImagePerSlice[sub, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    mask = (noisyImagesSubject * whiteMaskSubject)
    meanWhiteMatterInputImagePerSlice[sub, :] = meanPerSlice(mask.reshape(mask.shape[0], -1))
    crcInputImagePerSlice[sub, :] = crcValuePerSlice(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSlice[sub, :] = covValuePerSlice(noisyImagesSubject, greyMaskSubject)
    crcInputImagePerSubject[sub] = crcValuePerSubject(noisyImagesSubject, greyMaskSubject, whiteMaskSubject)
    covInputImagePerSubject[sub] = covValuePerSubject(noisyImagesSubject, greyMaskSubject)

    crcInputImageGlobal = np.mean(crcInputImagePerSubject[:])
    covInputImageGlobal = np.mean(covInputImagePerSubject[:])

    meanWhiteMatterInputImagePerSubject[sub] = meanPerSubject(meanWhiteMatterInputImagePerSlice[sub, :])
    meanGreyMatterInputImagePerSubject[sub] = meanPerSubject(meanGreyMatterInputImagePerSlice[sub, :])

    meanGreyMatterInputImageGlobal = meanPerSubject(meanGreyMatterInputImagePerSubject[:])
    meanWhiteMatterInputImageGlobal = meanPerSubject(meanWhiteMatterInputImagePerSubject[:])

    # METRICAS FILTROS + INPUT IMAGE
    for fil in range(0, len(filters)):
        numFilter = filters[fil]
        filter = (skimage.filters.gaussian(noisyImagesSubject, sigma=(numFilter / 2.35))).squeeze()

        if saveFilterOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(filter))
            image.SetSpacing(voxelSize)
            nameImage = 'Subject' + str(sub) +'_dose_'+str(lowDose_perc)+'_filter_'+str(fil)+'.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)

        mseFilterPerSlice[fil,sub,:]=mseValuePerSlice(filter,groundTruthSubject)

        mask = (filter * greyMaskSubject)
        meanGreyMatterFilterPerSlice[fil,sub,:] = meanPerSlice((mask.reshape(mask.shape[0], -1)))
        mask= (filter * whiteMaskSubject)
        meanWhiteMatterFilterPerSlice[fil,sub, :] = meanPerSlice((mask.reshape(mask.shape[0], -1)))

        crcFilterPerSlice[fil, sub, :] = crcValuePerSlice(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSlice[fil, sub, :] = covValuePerSlice(filter, greyMaskSubject)

        crcFilterPerSubject[fil, sub] = crcValuePerSubject(filter, greyMaskSubject, whiteMaskSubject)
        covFilterPerSubject[fil, sub] = covValuePerSubject(filter, greyMaskSubject)
        mseFilterPerSubject[fil, sub] = mseValuePerSubject(filter, groundTruthSubject)
        meanGreyMatterFilterPerSubject[fil, sub] = meanPerSubject(meanGreyMatterFilterPerSlice[fil, sub, :])
        meanWhiteMatterFilterPerSubject[fil, sub] = meanPerSubject(meanWhiteMatterFilterPerSlice[fil, sub, :])

        crcFilterGlobal[fil] = np.mean(crcFilterPerSubject[fil,:])
        covFilterGlobal[fil] = np.mean(covFilterPerSubject[fil,:])

        meanGreyMatterFilterGlobal[fil] = meanPerSubject(meanGreyMatterFilterPerSubject[fil, :])
        meanWhiteMatterFilterGlobal[fil] = meanPerSubject(meanWhiteMatterFilterPerSubject[fil, :])

# Input images + models
allModelsCrc = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsCov = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanGM = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanWM = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsCrcFilter = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsCovFilter = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanGMFilter = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanWMFilter = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMeanGMperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsMeanWMperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsCOVperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsCRCperSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))
allModelsMsePerSlice = np.zeros((len(modelFilenames), noisyImagesArray.shape[0], noisyImagesArray.shape[1]))
allModelsMsePerSubject = np.zeros((len(modelFilenames), noisyImagesArray.shape[0]))


# Resultados globales
allModelsCOVGlobal = np.zeros((len(modelFilenames), 1))
allModelsCRCGlobal = np.zeros((len(modelFilenames), 1))
allModelsMeanGMGlobal = np.zeros((len(modelFilenames), 1))
allModelsMeanWMGlobal = np.zeros((len(modelFilenames), 1))
allModelsMseGlobal = np.zeros((len(modelFilenames), 1))

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
        maxSliceSubject = maxSlice[sub, :].squeeze()
        print('Subject ', sub)

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
        if normalizeInput:
            ndaOutputModel = ndaOutputModel * maxSliceSubject[:,None,None]

        if saveModelOutputAsNiftiImage:
            image = sitk.GetImageFromArray(np.array(ndaOutputModel))
            image.SetSpacing(voxelSize)
            nameImage = 'Subject' + str(sub) + '_dose_' + str(lowDose_perc) + '_OutModel_' + modelName[-1]  + '.nii'
            save_path = os.path.join(pathSaveResults, nameImage)
            sitk.WriteImage(image, save_path)


        # Compute metrics for each slice:
        greyMaskedImage = (ndaOutputModel * greyMaskSubject)
        whiteMaskedImage = (ndaOutputModel * whiteMaskSubject)

        # Compute metrics for each subject all slices:
        allModelsMeanGM[contModel,sub,:] = meanPerSlice((greyMaskedImage.reshape(greyMaskedImage.shape[0], -1)))
        allModelsMeanWM[contModel, sub, :] = meanPerSlice((whiteMaskedImage.reshape(whiteMaskedImage.shape[0], -1)))
        allModelsCrc[contModel,sub,:] = crcValuePerSlice(ndaOutputModel, greyMaskSubject, whiteMaskSubject)
        allModelsCov[contModel,sub,:] = covValuePerSlice(ndaOutputModel, greyMaskSubject)

        # Compute metrics for all subject :
        allModelsCRCperSubject[contModel,sub] = crcValuePerSubject(ndaOutputModel,greyMaskSubject,whiteMaskSubject)
        allModelsCOVperSubject[contModel,sub] = covValuePerSubject(ndaOutputModel, greyMaskSubject)
        allModelsMeanGMperSubject[contModel, sub] = meanPerSubject(allModelsMeanGM[contModel, sub, :])
        allModelsMeanWMperSubject[contModel, sub] = meanPerSubject(allModelsMeanWM[contModel, sub, :])

    # Resultados globales

    allModelsCOVGlobal[contModel] = np.mean(allModelsCOVperSubject[contModel,:])
    allModelsCRCGlobal[contModel] = np.mean(allModelsCRCperSubject[contModel,:])
    allModelsMeanGMGlobal[contModel] = np.mean(allModelsMeanGMperSubject[contModel,:])
    allModelsMeanWMGlobal[contModel] = np.mean(allModelsMeanWMperSubject[contModel,:])

    contModel = contModel + 1


# Show plot
if showCovPlot == True:
    namesPlot = ['COV antes', 'COV modelos', 'COV filtros']
    showDataPlot(covInputImageGlobal,allModelsCOVGlobal,covFilterGlobal,filters,graphName = 'Cov',names=namesPlot)
    showDataPlot(covInputImagePerSubject, allModelsCOVperSubject, covFilterPerSubject, filters, graphName='Cov ',
                 names=namesPlot)

if showPerfilSlices == True:
    namesPlot = ['Mean antes', 'Mean model', 'Mean filtro']
    subjectPlot = 5
    meanFilter = meanGreyMatterFilterPerSlice[:, :, :].max(axis=2)
    meanOutModel = allModelsMeanGM[:, :, :].mean(axis=2)
    meanInputImage = meanGreyMatterInputImagePerSlice[:, :].max(axis=1)
    showDataPlot(meanGreyMatterInputImagePerSlice[subjectPlot, :] / meanInputImage[subjectPlot],
                 allModelsMeanGM[:, subjectPlot, :] / meanOutModel[:, subjectPlot, None],
                 meanGreyMatterFilterPerSlice[:, subjectPlot, :] / meanFilter[:, subjectPlot, None]
                 , filters, graphName='Mean Grey Matter',
                 names=namesPlot,saveFig = True, pathSave=pathSaveResults)

if saveDataCSV == True:
    saveDataCsv(meanGreyMatterFilterPerSlice, 'MeanGreyMatterFilterPerSlice.csv', pathSaveResults)

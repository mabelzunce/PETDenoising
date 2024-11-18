import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage import rotate
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

from cnn.python.testModels import pathSaveResults
from utils import covValuePerSlice
from utils import crcValuePerSlice
import math

from utils import saveDataCsv
from utils import crcValuePerSubject
from utils import covValuePerSubject
from utils import meanPerSubject


subjectName = []
# Importo base de datos ...
path = os.getcwd() + '/'
path = '../../../../../data/LowDoseRealData/CEUNIM2/'
pathSaveResults = '../../../../../data/LowDoseRealData/CEUNIM2/'
arraySubject = os.listdir(path)

subCount = 0

covNoisyImagePerSubject = np.zeros(len(arraySubject))
covFullDoseImagePerSubject = np.zeros(len(arraySubject))

crcNoisyImagePerSubject = np.zeros(len(arraySubject))
crcFullDoseImagePerSubject = np.zeros(len(arraySubject))

meanNoisyImagePerSubject = np.zeros(len(arraySubject))
meanFullDoseImagePerSubject = np.zeros(len(arraySubject))

for element in arraySubject:
    pathSubject = path+'/'+element
    #pathSubject = path + 'CP0042'
    subjectName.append(element)

    subjectMRI = sitk.ReadImage(pathSubject + '/T1_FS.nii')
    subjectMRI = sitk.GetArrayFromImage(subjectMRI)

    #subjectSeg = sitk.ReadImage(pathSubject + '/t1_robustfov_brain_seg.nii.gz')
    #subjectSeg = sitk.ReadImage(pathSubject + '/t1_robustfov_brain_pveseg.nii.gz')
    subjectSeg = sitk.ReadImage(pathSubject + '/aseg.nii')
    subjectSeg = sitk.GetArrayFromImage(subjectSeg)

    subjectPetHighDose = sitk.ReadImage(pathSubject + '/Registration_image_PET_T1.nii.gz')
    subjectPetHighDose = sitk.GetArrayFromImage(subjectPetHighDose)

    subjectPetLowDose = sitk.ReadImage(pathSubject + '/FRAME_1/PET_frame_to_MRI_1.nii')
    voxelSize_mm = subjectPetLowDose.GetSpacing()
    subjectPetLowDose = sitk.GetArrayFromImage(subjectPetLowDose)

    greyMatterSubject = np.where((subjectSeg == 42) | (subjectSeg == 3) | (subjectSeg == 8)| (subjectSeg == 47),1 , 0)
    whiteMatterSubject = np.where((subjectSeg == 41) | (subjectSeg == 2)| (subjectSeg == 7)| (subjectSeg == 46), 1, 0)

    covNoisyImagePerSubject[subCount] = covValuePerSubject(subjectPetLowDose, greyMatterSubject)
    covFullDoseImagePerSubject[subCount] = covValuePerSubject(subjectPetHighDose, greyMatterSubject)

    meanNoisyImagePerSubject[subCount] = meanPerSubject(subjectPetLowDose*greyMatterSubject)
    meanFullDoseImagePerSubject[subCount] = meanPerSubject(subjectPetHighDose*greyMatterSubject)

    crcNoisyImagePerSubject[subCount] = crcValuePerSubject(subjectPetLowDose, greyMatterSubject, whiteMatterSubject)
    crcFullDoseImagePerSubject[subCount] = crcValuePerSubject(subjectPetHighDose, greyMatterSubject, whiteMatterSubject)

    subCount = subCount+1

    # image = sitk.GetImageFromArray(np.array(greyMatterSubject))
    # image.SetSpacing(voxelSize_mm)
    # nameImage = 'trainGreyMatterMask.nii'
    # save_path = os.path.join(pathSubject, nameImage)
    # sitk.WriteImage(image, save_path)

    # import matplotlib.pyplot as plt
    #
    # # Crear una figura con 2 filas y 2 columnas de subplots
    # fig, axs = plt.subplots(2, 2)
    #
    # # Acceder y graficar en cada subplot
    # axs[0, 0].imshow(subjectMRI[100,:, :])
    # axs[0, 0].set_title("Resonancia")
    #
    # axs[0, 1].imshow(subjectSeg[100,:, :])
    # axs[0, 1].set_title("Segmentada")
    #
    # axs[1, 0].imshow(subjectPetHighDose[100,:, :])
    # axs[1, 0].set_title("PET 100%")
    #
    # axs[1, 1].imshow(subjectPetLowDose[100,:, :])
    # axs[1, 1].set_title("PET 5%")
    #
    # # Ajustar el espacio entre los subplots
    # plt.tight_layout()
    #
    # # Mostrar la figura
    # plt.show()

# Concatenación usando column_stack
covNoisyImagePerSubject = np.column_stack((subjectName, covNoisyImagePerSubject))
crcNoisyImagePerSubject = np.column_stack((subjectName, crcNoisyImagePerSubject))
meanNoisyImagePerSubject = np.column_stack((subjectName, meanNoisyImagePerSubject))
covFullDoseImagePerSubject = np.column_stack((subjectName, covFullDoseImagePerSubject))
crcFullDoseImagePerSubject = np.column_stack((subjectName, crcFullDoseImagePerSubject))
meanFullDoseImagePerSubject = np.column_stack((subjectName, meanFullDoseImagePerSubject))

saveDataCsv(covNoisyImagePerSubject, 'covNoisyImagePerSubject.csv', pathSaveResults)
saveDataCsv(crcNoisyImagePerSubject, 'crcNoisyImagePerSubject.csv', pathSaveResults)
saveDataCsv(meanNoisyImagePerSubject, 'meanNoisyImagePerSubject.csv', pathSaveResults)


saveDataCsv(covFullDoseImagePerSubject, 'covFullDoseImagePerSubject.csv', pathSaveResults)
saveDataCsv(crcFullDoseImagePerSubject, 'crcFullDoseImagePerSubject.csv', pathSaveResults)
saveDataCsv(meanFullDoseImagePerSubject, 'meanFullDoseImagePerSubject.csv', pathSaveResults)




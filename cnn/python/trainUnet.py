import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from datetime import datetime

from unet import Unet
from utils import imshow
from utils import MSE
import torch
import torch.nn as nn
import torchvision

############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn.
# If the dataset images are larger we will crop them.
imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.7
devSetRelSize = trainingSetRelSize-0.3


###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Read the training set:
noisyDataSet1_nii = sitk.ReadImage('./noisyDataSet1.nii')
ndaNoisyDataSet1 = sitk.GetArrayFromImage(noisyDataSet1_nii)

noisyDataSet2_nii = sitk.ReadImage('./noisyDataSet2.nii')
ndaNoisyDataSet2 = sitk.GetArrayFromImage(noisyDataSet2_nii)

groundTruth_nii = sitk.ReadImage('./groundTruth.nii')
ndaGroundTruth = sitk.GetArrayFromImage(groundTruth_nii)

# Check sizes:
if ndaNoisyDataSet1.shape != ndaGroundTruth.shape:
    print("The shape of noisy dataset {0} is different to the ground truth shape {1}.", img_noisyDataSet1.shape, img_groundTruth.shape)
    exit(-1)
# Size of each 2d image:
dataSetImageSize_voxels = ndaNoisyDataSet1.shape[1:3]
print("Data set original image size:",dataSetImageSize_voxels)

# Crop the image to be 256x256:
# Check sizes:
if ndaNoisyDataSet1.shape[1:2] != imageSize_voxels: #TODO: add the case were the input image is smaller.
    i_min = np.round((dataSetImageSize_voxels[0]-imageSize_voxels[0])/2).astype(int)
    i_max = np.round(dataSetImageSize_voxels[0]-(dataSetImageSize_voxels[0] - imageSize_voxels[0]) / 2).astype(int)
    j_min = np.round((dataSetImageSize_voxels[1] - imageSize_voxels[1]) / 2).astype(int)
    j_max = np.round(dataSetImageSize_voxels[1]-(dataSetImageSize_voxels[1] - imageSize_voxels[1]) / 2).astype(int)
    ndaNoisyDataSet1 =ndaNoisyDataSet1[:,i_min:i_max,j_min:j_max]
    ndaNoisyDataSet2 =ndaNoisyDataSet2[:,i_min:i_max,j_min:j_max]
    ndaGroundTruth =ndaGroundTruth[:,i_min:i_max,j_min:j_max]
    dataSetImageSize_voxels = ndaNoisyDataSet1.shape[1:3]
print("Data set cropped image size:",dataSetImageSize_voxels)

# Add the channel dimension for compatibility:
ndaNoisyDataSet1 = np.expand_dims(ndaNoisyDataSet1, axis=1)
ndaNoisyDataSet2 = np.expand_dims(ndaNoisyDataSet2, axis=0)
ndaGroundTruth = np.expand_dims(ndaGroundTruth, axis=1)
# Cast to float (the model expects a float):
ndaNoisyDataSet1 = ndaNoisyDataSet1.astype(np.float32)
ndaNoisyDataSet2 = ndaNoisyDataSet2.astype(np.float32)
ndaGroundTruth = ndaGroundTruth.astype(np.float32)

######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
# Get the number of images for the training and test data sets:
sizeFullDataSet = int(ndaNoisyDataSet1.shape[0])
sizeTrainingSet = int(np.round(sizeFullDataSet*trainingSetRelSize))
sizeDevSet = sizeFullDataSet-sizeTrainingSet
# Get random indices for the training set:
rng = np.random.default_rng()
indicesTrainingSet = rng.choice(int(sizeFullDataSet), int(sizeTrainingSet), replace=False)
indicesDevSet = np.delete(range(sizeFullDataSet), indicesTrainingSet)

# Create dictionaries with training sets:
trainingSet = dict([('input',ndaNoisyDataSet1[indicesTrainingSet,:,:,:]), ('output', ndaGroundTruth[indicesTrainingSet,:,:,:])])
devSet = dict([('input',ndaNoisyDataSet1[indicesDevSet,:,:,:]), ('output', ndaGroundTruth[indicesDevSet,:,:,:])])

print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(1,1)
inp = torch.rand(1, 1, 256, 256)
out = unet(inp)

##
print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape

##################################### U-NET TRAINING ############################################
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

# Number of  batches:
batchSize = 10
numBatches = np.round(trainingSet['input'].shape[0]/batchSize).astype(int)
# Show results every printStep batches:
printStep = 1
figImages, axs = plt.subplots(3, 1,figsize=(20,20))
figLoss, axLoss = plt.subplots(1, 1,figsize=(5,5))
# Show dev set loss every showDevLossStep batches:
showDevLossStep = 4
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
lossValuesTrainingSet = []
iterationNumbers = []
lossValuesDevSet = []
iterationNumbersForDevSet = []
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:])
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # Save loss values:
        lossValuesTrainingSet.append(loss.item())
        iterationNumbers.append(iter)
        # Evaluate dev set if it's the turn to do it:
        #if i % showDevLossStep == (showDevLossStep-1):
        #    outputsDevSet = unet(inputsDevSet)
        #    lossDevSet = criterion(outputsDevSet, gtDevSet)
        #    lossValuesDevSet.append(lossDevSet.item())
        #    iterationNumbersForDevSet.append(iter)
        # Show data it printStep
        if i % printStep == (printStep-1):    # print every printStep mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            # Show input images:
            plt.figure(figImages)
            plt.axes(axs[0])
            imshow(torchvision.utils.make_grid(inputs, normalize=True))
            axs[0].set_title('Input Batch {0}'.format(i))
            plt.axes(axs[1])
            imshow(torchvision.utils.make_grid(outputs, normalize=True))
            axs[1].set_title('Output Epoch {0}'.format(epoch))
            plt.axes(axs[2])
            imshow(torchvision.utils.make_grid(gt, normalize=True))
            axs[2].set_title('Ground Truth')
            # Show loss:
            plt.figure(figLoss)
            axLoss.plot(iterationNumbers, lossValuesTrainingSet)
            axLoss.plot(iterationNumbersForDevSet, lossValuesDevSet)
            plt.draw()
            plt.pause(0.0001)
        # Update iteration number:
        iter = iter + 1

print('Finished Training')

plt.pause(0)








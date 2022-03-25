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
import torch
import torch.nn as nn
import torchvision

# Size of the image we want to use in the cnn.
# If the dataset images are larger we will crop them.
imageSize_voxels = (256,256)

def MSE(img1, img2):
     cuadradoDeDif = ((img1 - img2) ** 2)
     suma = np.sum(cuadradoDeDif)
     cantPix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
     error = suma / cantPix
     return error

# Read the training set:
noisyDataSet1_nii = sitk.ReadImage('./noisyDataSet1.nii')
img_noisyDataSet1 = sitk.GetArrayFromImage(noisyDataSet1_nii)

noisyDataSet2_nii = sitk.ReadImage('./noisyDataSet2.nii')
img_noisyDataSet2 = sitk.GetArrayFromImage(noisyDataSet2_nii)

groundTruth_nii = sitk.ReadImage('./groundTruth.nii')
img_groundTruth = sitk.GetArrayFromImage(groundTruth_nii)

# Check sizes:
if img_noisyDataSet1.shape != img_groundTruth.shape:
    print("The shape of noisy dataset {0} is different to the ground truth shape {1}.", img_noisyDataSet1.shape, img_groundTruth.shape)
    exit(-1)
# Size of each 2d image:
dataSetImageSize_voxels = img_noisyDataSet1.shape[1:3]
print("Data set original image size:",dataSetImageSize_voxels)

# Crop the image to be 256x256:
# Check sizes:
if img_noisyDataSet1.shape[1:2] != imageSize_voxels: #TODO: add the case were the input image is smaller.
    i_min = np.round((dataSetImageSize_voxels[0]-imageSize_voxels[0])/2).astype(int)
    i_max = np.round(dataSetImageSize_voxels[0]-(dataSetImageSize_voxels[0] - imageSize_voxels[0]) / 2).astype(int)
    j_min = np.round((dataSetImageSize_voxels[1] - imageSize_voxels[1]) / 2).astype(int)
    j_max = np.round(dataSetImageSize_voxels[1]-(dataSetImageSize_voxels[1] - imageSize_voxels[1]) / 2).astype(int)
    img_noisyDataSet1 =img_noisyDataSet1[:,i_min:i_max,j_min:j_max]
    img_noisyDataSet2 =img_noisyDataSet2[:,i_min:i_max,j_min:j_max]
    img_groundTruth =img_groundTruth[:,i_min:i_max,j_min:j_max]
    dataSetImageSize_voxels = img_noisyDataSet1.shape[1:3]
print("Data set cropped image size:",dataSetImageSize_voxels)

# Add the channel dimension for compatibility:
img_noisyDataSet1 = np.expand_dims(img_noisyDataSet1, axis=1)
img_noisyDataSet2 = np.expand_dims(img_noisyDataSet2, axis=0)
img_groundTruth = np.expand_dims(img_groundTruth, axis=1)
print(img_noisyDataSet1.shape)
# Define a transform to convert
# the image to torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Convert the image to Torch tensor
#tensorNoisyDataSet1 = transform(img_noisyDataSet1)
#tensorNoisyDataSet2 = transform(img_noisyDataSet2)
#tensorGroundTruth = transform(img_groundTruth)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)
# Get a batch of the data.
img_noisyDataSet1 =img_noisyDataSet1[30:50,:,:,:]
img_noisyDataSet2 =img_noisyDataSet2[30:50,:,:,:]
img_groundTruth =img_groundTruth[30:50,:,:,:]
# Cast to float:
img_noisyDataSet1 = img_noisyDataSet1.astype(np.float32)
img_noisyDataSet2 = img_noisyDataSet2.astype(np.float32)
img_groundTruth = img_groundTruth.astype(np.float32)

# Create a UNET with one input and one output canal.
unet = Unet(1,1)

print(unet)
inp = torch.rand(1, 1, 256, 256)
out = unet(inp)

##
print('Test Unet. Output shape:',out.shape)
#tensorGroundTruth.shape

# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

# Number of  batches:
batchSize = 4
numBatches = np.round(img_noisyDataSet1.shape[0]/batchSize).astype(int)
# Show results every printStep batches:
printStep = 1
# Conjunto de entrenamiento, testeo y validacion
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(img_noisyDataSet1[i*batchSize:(i+1)*batchSize,:,:,:])
        gt = torch.from_numpy(img_groundTruth[i*batchSize:(i+1)*batchSize,:,:,:])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % printStep == (printStep-1):    # print every printStep mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')

# Show some results:
scaleForVisualization = 1.2*img_groundTruth.max()
imshow(torchvision.utils.make_grid(inputs), min=0, max=scaleForVisualization)






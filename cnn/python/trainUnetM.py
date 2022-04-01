import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy
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

import utils
from unet import Unet


unet = Unet()

# Importo base de datos

noisyDataSet1_nii = sitk.ReadImage('./noisyDataSet1.nii')
img_noisyDataSet1 = sitk.GetArrayFromImage(noisyDataSet1_nii)

noisyDataSet2_nii = sitk.ReadImage('./noisyDataSet2.nii')
img_noisyDataSet2 = sitk.GetArrayFromImage(noisyDataSet2_nii)

groundTruth_nii = sitk.ReadImage('./groundTruth.nii')
img_groundTruth = sitk.GetArrayFromImage(groundTruth_nii)

print("noisyDataSet1 shape:",img_noisyDataSet1.shape)
print("noisyDataSet2 shape:",img_noisyDataSet2.shape)
print("groundTruth shape:",img_groundTruth.shape)

# Reshape for training

img_noisyDataSet1 =img_noisyDataSet1[:,44:300,44:300]
img_noisyDataSet2 =img_noisyDataSet2[:,44:300,44:300]
img_groundTruth =img_groundTruth[:,44:300,44:300]


img_noisyDataSet1 = (np.expand_dims(img_noisyDataSet1, axis=-3)).astype(np.float32)
img_noisyDataSet2 = (np.expand_dims(img_noisyDataSet2, axis=-3)).astype(np.float32)
img_groundTruth = (np.expand_dims(img_groundTruth, axis=-3)).astype(np.float32)

print(img_noisyDataSet1.shape)
print(img_noisyDataSet2.shape)
print(img_groundTruth.shape)

# Conjunto de entrenamiento, testeo y validacion

train_noisyImage,valid_noisyImage,train_groundTruth,valid_groundTruth = train_test_split(img_noisyDataSet1, img_groundTruth, test_size=0.3)

# Create dictionaries with training sets:
trainingSet = dict([('input',train_noisyImage), ('output', train_groundTruth)])
validSet = dict([('input',valid_noisyImage), ('output', valid_groundTruth)])

print('Data set size. Training set: {0}. Valid set: {1}.'.format(trainingSet['input'].shape[0], validSet['input'].shape[0]))

# Entrenamiento #

# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

# Codigo entrenamiento Martin

# defino batches
batchSizeTrain = 4
batchSizeValid = 4
numBatchesTrain = np.round(trainingSet['input'].shape[0] / batchSizeTrain).astype(int)
numBatchesValid = np.round(validSet['input'].shape[0] / batchSizeValid).astype(int)

# Show dev set loss every showDevLossStep batches:
showDevLossStep = 4

printStep = 1
# figImages, axs = plt.subplots(3, 1,figsize=(20,20))
# figLoss, axLoss = plt.subplots(1, 1,figsize=(5,5))


# Train
loss_values = []
lossValuesTrainingSet = []
iterationNumbers = []
lossValuesDevSet = []
iterationNumbersForDevSet = []

lossValuesEpoch = []

iter = 0

EPOCHS = 1

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0

    unet.train(True)
    for i in range(numBatchesTrain):
        # get the inputs

        inputs = torch.from_numpy(trainingSet['input'][i * batchSizeTrain:(i + 1) * batchSizeTrain, :, :, :])
        gt = torch.from_numpy(trainingSet['output'][i * batchSizeTrain:(i + 1) * batchSizeTrain, :, :, :])

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

        if i % printStep == (printStep - 1):  # print every printStep mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

            # Show input images:
            # plt.figure(figImages)
            # plt.axes(axs[0])
            # imshow(torchvision.utils.make_grid(inputs, normalize=True))
            # axs[0].set_title('Input Batch {0}'.format(i))
            # plt.axes(axs[1])
            # imshow(torchvision.utils.make_grid(outputs, normalize=True))
            # axs[1].set_title('Output Epoch {0}'.format(epoch))
            # plt.axes(axs[2])
            # imshow(torchvision.utils.make_grid(gt, normalize=True))
            # axs[2].set_title('Ground Truth')
            # Show loss:
            # plt.figure(figLoss)
            # axLoss.plot(iterationNumbers, lossValuesTrainingSet)
            # axLoss.plot(iterationNumbersForDevSet, lossValuesDevSet)
            # plt.draw()
            # plt.pause(0.0001)

            # Update iteration number:
        iter = iter + 1

    lossValuesEpoch.append(lossValuesTrainingSet/numBatchesTrain)
    unet.train(False)
    running_vloss = 0.0


    for i in range(numBatchesValid):
        print(i)

        vinputs = torch.from_numpy(validSet['input'][i * batchSizeValid:(i + 1) * batchSizeValid, :, :, :])
        vgt = torch.from_numpy(validSet['output'][i * batchSizeValid:(i + 1) * batchSizeValid, :, :, :])

        voutputs = unet(vinputs)
        vloss = criterion(voutputs, vgt)
        vloss.backward()
        running_vloss += vloss

        lossValuesDevSet.append(vloss.item())


    avg_vloss = numpy.mean(lossValuesDevSet)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    print('LOSS train {} valid {}'.format(lossValuesTrainingSet[-1], avg_vloss))

print('Finished Training')

plt.pause(0)

# Test set

inputsTestSet = torch.from_numpy(trainingSet['input'][:,:,:,:])
inputsTestSet = torchvision.transforms.functional.rotate(inputsTestSet,15)


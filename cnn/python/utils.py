import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def imshow(img, min=0, max=1):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = min, vmax = max)

def MSE(img1, img2, cantPix = None):
    cuadradoDeDif = ((img1 - img2) ** 2)
    suma = np.sum(cuadradoDeDif)
    if cantPix == 'None':
        cantPix = img1.shape[2] * img1.shape[1]  # img1 and 2 should have same shape
    else:
        cantPix = cantPix
    error = suma / cantPix
    return error

def trainModel(model, trainSet, validSet, criterion, optimizer, num_batch, epochs, pre_trained=False):
    # defino batches

    best_vloss = 1000000000

    batchSizeTrain = num_batch
    batchSizeValid = num_batch
    numBatchesTrain = np.round(trainSet['input'].shape[0] / batchSizeTrain).astype(int)
    numBatchesValid = np.round(trainSet['input'].shape[0] / batchSizeValid).astype(int)

    # Show dev set loss every showDevLossStep batches:
    showDevLossStep = 4

    printStep = 1
    # figImages, axs = plt.subplots(3, 1,figsize=(20,20))
    # figLoss, axLoss = plt.subplots(1, 1,figsize=(5,5))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Train
    loss_values = []
    lossValuesTrainingSet = []
    iterationNumbers = []
    lossValuesDevSet = []
    iterationNumbersForDevSet = []
    lossValuesTrainingSetEpoch = []

    lossValuesEpoch = []
    lossValuesDevSetAllEpoch = []

    iter = 0

    EPOCHS = epochs

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        lossValuesTrainingSetEpoch = []
        lossValuesDevSetEpoch = []

        running_loss = 0.0

        model.train(True)
        for i in range(numBatchesTrain):
            # get the inputs

            inputs = torch.from_numpy(trainSet['input'][i * batchSizeTrain:(i + 1) * batchSizeTrain, :, :, :])
            gt = torch.from_numpy(trainSet['output'][i * batchSizeTrain:(i + 1) * batchSizeTrain, :, :, :])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # Save loss values:
            lossValuesTrainingSet.append(loss.item())
            lossValuesTrainingSetEpoch.append(loss.item())
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

        lossValuesEpoch.append(np.mean(lossValuesTrainingSetEpoch))
        model.train(False)
        running_vloss = 0.0

        for i in range(numBatchesValid):
            print(i)

            vinputs = torch.from_numpy(validSet['input'][i * batchSizeValid:(i + 1) * batchSizeValid, :, :, :])
            vgt = torch.from_numpy(validSet['output'][i * batchSizeValid:(i + 1) * batchSizeValid, :, :, :])

            voutputs = model(vinputs)
            vloss = criterion(voutputs, vgt)
            vloss.backward()
            running_vloss += vloss

            lossValuesDevSet.append(vloss.item())

            lossValuesDevSetEpoch.append(vloss.item())

        avg_vloss = np.mean(lossValuesDevSet)
        lossValuesDevSetAllEpoch.append(np.mean(lossValuesDevSetEpoch))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'modelDataSet2_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        print('LOSS train {} valid {}'.format(lossValuesEpoch[-1], lossValuesDevSetAllEpoch[-1]))
        # CALCULAR PROMEDIO DE TODOS O VARIOS BATCH

    print('Finished Training')

    return lossValuesTrainingSet, lossValuesEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch

def reshapeDataSet(dataset):

    dataset = dataset[:, 44:300, 44:300]
    dataset = (np.expand_dims(dataset, axis=-3)).astype(np.float32)

    return dataset
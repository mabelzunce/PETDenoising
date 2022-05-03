import torch
import torchvision
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from datetime import datetime

def imshow(img, min=0, max=1):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = min, vmax = max)
    return

def showGridNumpyImg(inputs, outputs, gt, plotTitle):
    vmin = gt[0].min()
    vmax = gt[0].max()

    cantImg = len(inputs)

    totalImg = [inputs + outputs + gt]

    fig = plt.figure(figsize=(4, 4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, cantImg),
                     axes_pad=0.1,
                     )

    labels = ['Input', 'Output', 'Ground Truth']

    i = 0
    cont = 0
    for ax, im in zip(grid, totalImg[0]):
        img = np.transpose(im[0,:,:,:], (1, 2, 0))
        ax.imshow(img, vmin = vmin, vmax = vmax)
        if (i == 0) or (i%cantImg) == 0:
            ax.set_ylabel(labels[cont])
            cont = cont + 1
        i = i + 1

    grid.axes_all[1].set_title(plotTitle)
    plt.show(block=False)
    name = plotTitle+'.png'
    plt.savefig(name)

    return

def MSE(img1, img2, cantPixels = None):
    cuadradoDeDif = ((img1 - img2) ** 2)
    suma = np.sum(cuadradoDeDif)
    if cantPixels == None:
        cantPix = img1.shape[2] * img1.shape[1]  # img1 and 2 should have same shape
    else:
        cantPix = cantPixels
    error = suma / cantPix
    return error

def trainModel(model, trainSet, validSet, criterion, optimizer, num_batch, epochs, pre_trained = False):
    # defino batches

    best_vloss = 1000000000

    batchSizeTrain = num_batch
    batchSizeValid = num_batch
    numBatchesTrain = np.round(trainSet['input'].shape[0] / batchSizeTrain).astype(int)
    numBatchesValid = np.round(validSet['input'].shape[0] / batchSizeValid).astype(int)

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

        avg_vloss = np.mean(lossValuesDevSetEpoch)
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

    "Reshape a TotalImagesx1x256x256"

    dataset = dataset[:, 44:300, 44:300]
    dataset = (np.expand_dims(dataset, axis=-3)).astype(np.float32)

    return dataset

def torchToNp(dataTorch) :
    "Pasamos de toch a imagen"
    img = torch.unsqueeze(dataTorch, dim=0)
    img = (img).detach().numpy()
    return img

def testModelSlice(model, inputsDataSet):

    inputs = inputsDataSet
    inputs = torch.unsqueeze(inputs, dim=0)
    out = model(inputs)

    return out

def mseAntDspModelTorchSlice(inputs,outputs, groundTruth, cantPixels = None):
    '''Devuelve el MSE de una imagen
    antes y dsp de pasar por el modelo'''

    # Antes

    img1 = torchToNp(inputs)
    img2 = torchToNp(groundTruth)

    mseAntes = (MSE(img1[0, :, :], img2[0, :, :],cantPixels))

    # Dsp
    img3 = torchToNp(outputs)
    mseDespues = (MSE(img3[0, :, :], img2[0, :, :],cantPixels))

    return mseAntes, mseDespues

def obtenerMask(img, num_mask) :
    return ((img == num_mask) * 1.0)

def saveNumpyAsNii(np_img,name):
    '''Recibe numpy array de 4 dimensiones
    y la guarda en formato .Nii'''
    image = sitk.GetImageFromArray(np_img[0, :, :])
    img_name = name + '.nii'
    sitk.WriteImage(image, img_name)
    return


def getTestOneModelOneSlices(model, inputs, groundTruth, display = 'False' ,mse='False', mseGrey='False', greyMatterValue=None,
                             mseWhite='False', whiteMatterValue=None):
    '''Para usar esta funcion inputs y
    grounTruth deben ser Tensores
    inputs es solo un slice'''

    model = model
    outModel = testModelSlice(model, inputs)

    if mse == 'True':
        mseBef, mseAft = mseAntDspModelTorchSlice(inputs, outModel[0, :, :, :], groundTruth)
        if display == 'True':
            print('MSE:')
            print('MSE antes de pasar por la red', mseBef)
            print('MSE dsp de pasar por la red', mseAft)

        return outModel, mseBef, mseAft

    if mseGrey == 'True':
        greyMatterValue = greyMatterValue

        greyMaskMatter = obtenerMask(groundTruth, greyMatterValue)

        greyMatterNoisyAntes = inputs * greyMaskMatter
        greyMatterNoisyDsp = outModel * greyMaskMatter

        groundTruthGreyMatter = groundTruth * greyMaskMatter
        cantPix = np.count_nonzero(groundTruthGreyMatter)

        mseBefGrey, mseAftGrey = mseAntDspModelTorchSlice(greyMatterNoisyAntes, greyMatterNoisyDsp,
                                                          groundTruthGreyMatter, cantPix)
        if display == 'True':
            print('MATERIA GRIS')
            print('MSE antes de pasar por la red', mseBefGrey)
            print('MSE dsp de pasar por la red', mseAftGrey)

        return outModel, mseBefGrey, mseAftGrey

    if mseWhite == 'True':
        whiteMatterValue = whiteMatterValue

        whiteMaskMatter = obtenerMask(groundTruth, whiteMatterValue)

        whiteMatterNoisyAntes = inputs * whiteMaskMatter
        whiteMatterNoisyDsp = outModel * whiteMaskMatter

        cantPix = np.count_nonzero(whiteMaskMatter)

        groundTruthWhiteMatterNoisy = groundTruth * whiteMaskMatter

        mseBefWhite, mseAftWhite = mseAntDspModelTorchSlice(whiteMatterNoisyAntes, whiteMatterNoisyDsp,

                                                            groundTruthWhiteMatterNoisy, cantPix)

        if display == 'True':
            print('MATERIA BLANCA')
            print('MSE antes de pasar por la red', mseBefWhite)
            print('MSE dsp de pasar por la red', mseAftWhite)

        return outModel, mseBefWhite, mseAftWhite

    else:

        return outModel

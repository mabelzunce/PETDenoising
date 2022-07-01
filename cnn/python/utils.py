import torch
import torchvision
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from datetime import datetime

def imshow(img, min=0, max=1):
    img = img
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

def showSubplots(img, title):
    cantImages = int(img.shape[0]**(1/2))
    if cantImages > 5:
        cantImages = 5
    rows = cantImages
    cols = cantImages
    img_count = 0
    fig, axes = plt.subplots(nrows=cantImages, ncols=cantImages)
    for ax in axes.flat:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if img_count < img.shape[0]:
            im = ax.imshow(img[img_count, 0, :, :],cmap = 'gray')
            img_count = img_count + 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle(title)
    plt.show(block=False)
    name = title + '.png'
    plt.savefig(name)
    plt.close()

def MSE(img1, img2, cantPixels = None):
    cuadradoDeDif = ((img1 - img2) ** 2)
    suma = np.sum(cuadradoDeDif)
    if cantPixels == None:
        cantPix = img1.shape[2] * img1.shape[1]  # img1 and 2 should have same shape
    else:
        cantPix = cantPixels
    error = suma / cantPix
    return error

def trainModel(model, trainSet, validSet, criterion, optimizer, num_batch, epochs, pre_trained = False, save = True, name = None):
    # defino batches
    # Return
    # lossValuesTrainingSet: loss por batch para trainSet
    # lossValueTrainingSetAllEpoch: loss por epoca para trainSet
    # lossValuesDevSet: loss por batch para validSet
    # lossValuesDevSetAllEpoch: loss por epoca para validSet

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
    iterationNumbers = []

    loss_values = []
    lossValuesTrainingSet = []


    lossValuesDevSet = []
    iterationNumbersForDevSet = []
    lossValuesTrainingSetEpoch = []

    lossValuesDevSetAllEpoch = []
    lossValueTrainingSetAllEpoch = []

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

                #x = np.arange(0, len(lossValuesTrainingSet))
                #y1 = lossValuesTrainingSet
                #plt.plot(x, y1)
                #plt.title('Batch train set')
                #plt.draw()
                #plt.pause(0.0001)

            iter = iter + 1

        lossValueTrainingSetAllEpoch.append(np.mean(lossValuesTrainingSetEpoch))
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
            model_path = 'modelDataSet6_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)


        print('LOSS train {} valid {}'.format(lossValueTrainingSetAllEpoch[-1], lossValuesDevSetAllEpoch[-1]))

        if (save == True) and (epoch%5 == 0):

            nameArch = 'lossValuesTrainingSetBatch'+name+'.xlsx'
            df = pd.DataFrame(lossValuesTrainingSet)
            df.to_excel(nameArch)

            nameArch = 'lossValuesTrainingSetEpoch' + name + '.xlsx'
            df = pd.DataFrame(lossValueTrainingSetAllEpoch)
            df.to_excel(nameArch)

            nameArch = 'lossValuesDevSetBatch' + name + '.xlsx'
            df = pd.DataFrame(lossValuesDevSet)
            df.to_excel(nameArch)

            nameArch = 'lossValuesDevSetEpoch' + name + '.xlsx'
            df = pd.DataFrame(lossValuesDevSetAllEpoch)
            df.to_excel(nameArch)

            x = np.arange(0, len(lossValueTrainingSetAllEpoch))
            y1 = lossValueTrainingSetAllEpoch
            y2 = lossValuesDevSetAllEpoch
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.title('Epochs')
            plt.draw()
            plt.pause(0.0001)

    print('Finished Training')

    return lossValuesTrainingSet, lossValueTrainingSetAllEpoch, lossValuesDevSet, lossValuesDevSetAllEpoch

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

    #torch.Tensor.numpy

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


def getTestOneModelOneSlices(inputs, outModel,groundTruth, display = 'False' ,mse='False', mseGrey='False', maskGrey = None, greyMatterValue=None,
                             mseWhite='False', maskWhite= None, whiteMatterValue=None):

    if mse == 'True':
        mseBef, mseAft = mseAntDspModelTorchSlice(inputs, outModel, groundTruth)
        if display == 'True':
            print('MSE:')
            print('MSE antes de pasar por la red', mseBef)
            print('MSE dsp de pasar por la red', mseAft)

        return outModel, mseBef, mseAft

    if mseGrey == 'True':
        greyMatterValue = greyMatterValue

        if maskGrey == None:
            greyMaskMatter = obtenerMask(groundTruth, greyMatterValue)
        else:
            greyMaskMatter = maskGrey

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

        if maskWhite == None :
            whiteMaskMatter = obtenerMask(groundTruth, whiteMatterValue)
        else:
            whiteMaskMatter = maskWhite

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

def covValue(img, maskGrey):
    '''
    Se calcula a partir del std y mean en materia gris
    '''
    materiaGris = img * maskGrey

    materiaGris = (torch.Tensor.numpy(materiaGris)).flatten()

    materiaGris = materiaGris[materiaGris != 0.0]

    meanMateriaGris = np.mean(materiaGris)
    stdMateriaGris = np.std(materiaGris)

    cov= stdMateriaGris / meanMateriaGris

    return cov

def crcValue(img, maskGrey, maskWhite):
    '''
    mean materia gris/ mean materia blanca
    '''

    materiaBlanca = maskWhite * img
    materiaGris = maskGrey * img

    materiaGris = (torch.Tensor.numpy(materiaGris)).flatten()
    materiaBlanca = (torch.Tensor.numpy(materiaBlanca)).flatten()

    materiaGris  = materiaGris [materiaGris  != 0.0]
    materiaBlanca = materiaBlanca[materiaBlanca != 0.0]

    meanMateriaGris = np.mean(materiaGris)
    meanMateriaBlanca = np.mean(materiaBlanca)

    crc = meanMateriaGris / meanMateriaBlanca

    return crc

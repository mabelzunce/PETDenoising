import nibabel as nb
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from utils import trainModel
from utils import reshapeDataSet
from utils import showSubplots
#from unetM import Unet
#import unetM

from unet import UnetWithResidual

path = os.getcwd()

# Model:
learning_rate=0.00005
normalizeInput = True
nameModel = 'UnetWithResidual_MSE_lr{0}_AlignTrue'.format(learning_rate)
if normalizeInput:
    nameModel = nameModel + '_norm'
pathModel = '../../../Results/' + nameModel + '/models/'

nameThisNet = 'Unet5Layers_MSE_lr{0}_AlignTrue'.format(learning_rate)


model = UnetWithResidual(1,1)
modelsPath = pathModel + 'UnetWithResidual_MSE_lr5e-05_AlignTrue_norm_20220715_191324_27_best_fit' #nameModel + str(epoch) + '_best_fit'
#modelsPath = path+'/ModeloUnetResidualNoNorm/UnetWithResidual_MSE_lr5e-05_22_best_fit'
model.load_state_dict(torch.load(modelsPath, map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('bestModelDataSet3_6'))

#nameModel = 'UnetWithResidual_MSE_lr5e-05_33_best_fit'

# Data:
path = '../../data/BrainWebSimulations/'
pathGroundTruth = path+'/100'
arrayGroundTruth = os.listdir(pathGroundTruth)

pathNoisyDataSet = path+'/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)

#pathGreyMask = path+'/Phantoms/'
#arrayGreyMask= os.listdir(pathGreyMask)

#pathWhiteMask = path+'/Phantoms/'
#arrayWhiteMask= os.listdir(pathWhiteMask)

#calculo metricas por sujeto...
# leo el fantoma, el ground truth y las mascaras por sujeto

nameGroundTruth=[]
groundTruthArray = []

for element in arrayGroundTruth:
    pathGroundTruthElement = pathGroundTruth+'/'+element
    groundTruth = sitk.ReadImage(pathGroundTruthElement)
    groundTruth = sitk.GetArrayFromImage(groundTruth)
    groundTruth = reshapeDataSet(groundTruth)
    groundTruthArray.append(groundTruth)
    nameGroundTruth.append(element[23:-4])

nameNoisyDataSet=[]
noisyImagesArray = []
for element in arrayNoisyDataSet:
    pathNoisyDataSetElement = pathNoisyDataSet+'/'+element
    noisyDataSet = sitk.ReadImage(pathNoisyDataSetElement)
    noisyDataSet = sitk.GetArrayFromImage(noisyDataSet)
    noisyDataSet = reshapeDataSet(noisyDataSet)
    noisyImagesArray.append(noisyDataSet)
    nameNoisyDataSet.append(element[21:-4])

# paso al modelo y evaluo resultados
noisyImagesArray = np.array(noisyImagesArray)
groundTruthArray = np.array(groundTruthArray)

# Get the maximum value per slice:
maxSlice = noisyImagesArray[:, :, :, :].max(axis=4).max(axis=3)
maxSliceGroundTruth = groundTruthArray[:, :, :, :].max(axis=4).max(axis=3)
# Normalize the input if necessary:
if normalizeInput:
    noisyImagesArray = noisyImagesArray/maxSlice[:,:,:,None,None]
    noisyImagesArray = np.nan_to_num(noisyImagesArray)

noisyImagesArray = torch.from_numpy(noisyImagesArray)
groundTruthArray = torch.from_numpy(groundTruthArray)



subject = 1
sliceNro = 32

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



activation = {}

model.down1.register_forward_hook(get_activation('down1'))
model.down4.register_forward_hook(get_activation('down4'))

model.up4.register_forward_hook(get_activation('up4'))
model.outc.register_forward_hook(get_activation('outc'))
model.add_res.register_forward_hook(get_activation('add_res'))


inputsModel = torch.unsqueeze(noisyImagesArray[subject,sliceNro,:,:,:], dim=0)
output = model(inputsModel)

showSubplots(activation['up4'][0,:,:,:], 'LayerUp4_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
showSubplots(activation['down1'][0,:,:,:], 'LayerDown1_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
showSubplots(activation['down4'][0,:,:,:], 'LayerDown4_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
showSubplots(activation['outc'][0,:,:,:], 'LayerOutc_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
showSubplots(activation['add_res'][0,:,:,:], 'LayerAdd_res_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)

showSubplots(torch.Tensor.numpy(inputsModel[:,:, :, :]).squeeze(axis = 1), 'Input_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
showSubplots((torch.Tensor.numpy(output[:,:, :, :].detach())).squeeze(axis = 1), 'Output_Subject'+str(subject)+'_slice'+str(sliceNro)+'Model_'+nameModel)
plt.show(block=True)
#model = UnetWithResidual(1,1)
#modelsPath = path+'/ModeloUnetResidualUno/UnetWithResidual_MSE_lr5e-05_33_best_fit'
#model.load_state_dict(torch.load(modelsPath, map_location=torch.device('cpu')))

#modelDT3 = Unet()
#modelDT3.load_state_dict(torch.load('bestModelDataSet3_6'))

# guardo los kernels...
#model_weights = [] # we will save the conv layer weights in this list
#conv_layers_down = [] # we will save the 49 conv layers in this list
#conv_layers_up = []
# counter to keep count of the conv layers
#counter = 0
# append all the conv layers and their respective weights to the list
#model_children = list(model.children())

#for i in range(len(model_children)):
    #if type(model_children[i]) == unetM.DownConv:
    #if type(model_children[i]) == UnetWithResidual.Down:
        #for j in range(len(model_children[i].DownLayer)):
        #for j in range(len(model_children[i].conv)):
            #child = model_children[i].DownLayer[j]
            #child = model_children[i].conv[j]
            #if type(child) == nn.Conv2d:
                #counter += 1
                #model_weights.append(child.weight)
                #conv_layers_down.append(child)

    #if type(model_children[i]) == unetM.UpConv:
    #if type(model_children[i]) == UnetWithResidual.Up:
        #for j in range(len(model_children[i].UpConv.DownLayer)):
        #for j in range(len(model_children[i].Up.Down)):
            #child = model_children[i].UpConv.DownLayer[j]
            #child = model_children[i].Up.Down[j]
            #if type(child) == nn.Conv2d:
                #counter += 1
                #model_weights.append(child.weight)
                #conv_layers_up.append(child)


#for n in range(0,len(model_weights)):
    #numLayer = 1+int(n/2)
    #numConv = 0+(n%2)
    #    showSubplots(model_weights[n].detach(), f"Kernels_layer{numLayer}_conv{numConv}.png")

#inputs = torch.unsqueeze(noisyImagesArray[subject,sliceNro,:,:,:], dim=0)
#results = [conv_layers_down[0](inputs)]
#for i in range(1, len(conv_layers_down)):
    # pass the result from the last layer to the next layer
#    results.append(conv_layers_down[i](results[-1]))
# make a copy of the `results`
#outputs = results

#for num_layer in range(len(outputs)):
#    layer_viz = outputs[num_layer][0, :, :, :]
#    layer_viz = layer_viz.data
#    print(layer_viz.size())
#    numLayer = 1 + int(num_layer / 2)
#    numConv = 0 + (num_layer % 2)
#    showSubplots(torch.unsqueeze(layer_viz.detach(),dim=1), f"Outputs_layer{numLayer}_conv{numConv}.png")



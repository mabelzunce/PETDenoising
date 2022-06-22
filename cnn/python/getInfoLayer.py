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
from utils import showSubplots, plot_weights
from unetM import Unet
import unetM

path = os.getcwd()
pathGroundTruth = path+'/groundTruth/100'
arrayGroundTruth = os.listdir(pathGroundTruth)

pathNoisyDataSet = path+'/noisyDataSet/5'
arrayNoisyDataSet= os.listdir(pathNoisyDataSet)

pathGreyMask = path+'/PhantomsGreyMask'
arrayGreyMask= os.listdir(pathGreyMask)

pathWhiteMask = path+'/PhantomsWhiteMask'
arrayWhiteMask= os.listdir(pathWhiteMask)

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

noisyImagesArray = torch.from_numpy(noisyImagesArray)
groundTruthArray = torch.from_numpy(groundTruthArray)

subject = 5
sliceNro = 1

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = Unet()
model.load_state_dict(torch.load('bestModelDataSet3_6'))

activation = {}

model.Layer1Down.register_forward_hook(get_activation('Layer1Down'))
model.Layer2Down.register_forward_hook(get_activation('Layer2Down'))
model.Layer3Down.register_forward_hook(get_activation('Layer3Down'))
model.Layer4Down.register_forward_hook(get_activation('Layer4Down'))
model.Layer5Down.register_forward_hook(get_activation('Layer5Down'))

model.Middle.register_forward_hook(get_activation('Middle'))

model.Layer1Up.register_forward_hook(get_activation('Layer1Up'))
model.Layer2Up.register_forward_hook(get_activation('Layer2Up'))
model.Layer3Up.register_forward_hook(get_activation('Layer3Up'))
model.Layer4Up.register_forward_hook(get_activation('Layer4Up'))
model.Layer5Up.register_forward_hook(get_activation('Layer5Up'))

inputsModel = torch.unsqueeze(noisyImagesArray[subject,sliceNro,:,:,:], dim=0)
output = model(inputsModel)

showSubplots(torch.unsqueeze(activation['Layer1Down'][0,:,:,:], dim=1), 'Layer 1 DOWN')
showSubplots(torch.unsqueeze(activation['Layer2Down'][0,:,:,:], dim=1), 'Layer 2 DOWN')
showSubplots(torch.unsqueeze(activation['Layer3Down'][0,:,:,:], dim=1), 'Layer 3 DOWN')
showSubplots(torch.unsqueeze(activation['Layer4Down'][0,:,:,:], dim=1), 'Layer 4 DOWN')
showSubplots(torch.unsqueeze(activation['Layer5Down'][0,:,:,:], dim=1), 'Layer 5 DOWN')

showSubplots(torch.unsqueeze(activation['Middle'][0,:,:,:], dim=1), 'Middle')

showSubplots(torch.unsqueeze(activation['Layer1Up'][0,:,:,:], dim=1), 'Layer 1 UP')
showSubplots(torch.unsqueeze(activation['Layer2Up'][0,:,:,:], dim=1), 'Layer 2 UP')
showSubplots(torch.unsqueeze(activation['Layer3Up'][0,:,:,:], dim=1), 'Layer 3 UP')
showSubplots(torch.unsqueeze(activation['Layer4Up'][0,:,:,:], dim=1), 'Layer 4 UP')
showSubplots(torch.unsqueeze(activation['Layer5Up'][0,:,:,:], dim=1), 'Layer 5 UP')

image = sitk.GetImageFromArray((torch.Tensor.numpy(inputsModel[0,0, :, :])))
sitk.WriteImage(image, f"Input_Subject{subject}_slice{sliceNro}.nii")

image = sitk.GetImageFromArray((torch.Tensor.numpy(output[0,0, :, :])))
sitk.WriteImage(image, f"Output_Subject{subject}_slice{sliceNro}.nii")

modelDT3 = Unet()
modelDT3.load_state_dict(torch.load('bestModelDataSet3_6'))

# guardo los kernels...
model_weights = [] # we will save the conv layer weights in this list
conv_layers_down = [] # we will save the 49 conv layers in this list
conv_layers_up = []
# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
model_children = list(modelDT3.children())

for i in range(len(model_children)):
    if type(model_children[i]) == unetM.DownConv:
        for j in range(len(model_children[i].DownLayer)):
            child = model_children[i].DownLayer[j]
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers_down.append(child)

    if type(model_children[i]) == unetM.UpConv:
        for j in range(len(model_children[i].UpConv.DownLayer)):
            child = model_children[i].UpConv.DownLayer[j]
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers_up.append(child)


for n in range(0,len(model_weights)):
    numLayer = 1+int(n/2)
    numConv = 0+(n%2)
    showSubplots(model_weights[n].detach(), f"Kernels_layer{numLayer}_conv{numConv}.png")

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



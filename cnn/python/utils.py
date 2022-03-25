import torch
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, min=0, max=1):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = min, vmax = max)


def MSE(img1, img2):
    cuadradoDeDif = ((img1 - img2) ** 2)
    suma = np.sum(cuadradoDeDif)
    cantPix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
    error = suma / cantPix
    return error
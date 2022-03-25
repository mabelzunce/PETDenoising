import torch
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, min=0, max=1):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = min, vmax = max)
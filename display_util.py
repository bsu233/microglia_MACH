"""
Creates stacked image with gradient of transparent reds
"""
# from https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

def DefineAlphaCmap(daMap="Reds"):

    # Choose colormap
    if daMap is "Reds":
      cmap = plt.cm.Reds
    else:
      raise RuntimeError("Need to generalize")
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    return my_cmap

def StackGrayRedAlpha(img1,img2,alpha=1.):
    my_cmap = DefineAlphaCmap()
    plt.imshow(img1,cmap="gray")
    plt.imshow(img2,cmap=my_cmap,alpha=alpha)

def ExampleImage():
    img1=np.array(np.random.rand(100*100)*2e7,dtype=np.uint8)
    img1 = np.reshape(img1,[100,100])

    img2 = np.outer(np.linspace(0,100,101),np.ones(100))

    StackGrayRedAlpha(img1,img2)


#ExampleImage()

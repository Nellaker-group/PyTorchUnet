import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter

# for creating a gaussian kernel - that can be used a kernel
# from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# for applying kernel to numpy array - CHECK IF INDEED WORKING
# from https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array
def blur(a,filter="boxlur"):
    assert filter in ["boxblur","gaussblur","edge"]
    if filter == "boxblur":
        kernel = np.array([[1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0]])
        kernel = kernel / np.sum(kernel)
    elif filter == "gaussblur":
        kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
        kernel = kernel / np.sum(kernel)
    else:
        kernel = np.array([[1.0,0.0,-1.0], [0.0,0.0,0.0], [-1.0,0.0,1.0]])
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)
    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def augmenterTmp(image):

    choice=np.random.randint(0,9)

    if choice == 0:
        # flips array left right (vertically)
        image=np.fliplr(image)
        return(image)
    elif choice == 1:
        # flips array up down (horizontically)
        image=np.flipud(image)
        return(image)
    elif choice == 2:
        # moving each element one place clockwise
        image=np.rot90(image, k=1, axes=(1,0))
        return(image)
    elif choice == 3:
        # moving each element one place counter clockwise
        image=np.rot90(image, k=1, axes=(0,1))
        return(image)
    elif choice == 4:
        # gaussian noise
        noise = np.random.normal(0,1,(1024,1024))
        return(image+noise)
        # ...
    elif choice == 5:
        # manual blurring using numpy
        return(blur(image))
        # ...
    elif choice == 6:
        # scipy gaussian blurring
        return(gaussian_filter(image, sigma=1))
        # ...
    else:
        return(image)
    
# I have to add .copy() to make it work with transforming it to a tensor
# see this for more:
# https://stackoverflow.com/questions/20843544/np-rot90-corrupts-an-opencv-image
def augmenter(image,mask):
    # half of the time it augments
    choice=np.random.randint(0,10)
    if choice == 0:
        # flips array left right (vertically)
        return(np.fliplr(image).copy(),np.fliplr(mask).copy())
    elif choice == 1:
        # flips array up down (horizontically)
        return(np.flipud(image).copy(),np.flipud(mask).copy())
    elif choice == 2:
        # moving each element one place clockwise
        return(np.rot90(image, k=1, axes=(1,0)).copy(),np.rot90(mask, k=1, axes=(1,0)).copy())
    elif choice == 3:
        # moving each element one place counter clockwise
        return(np.rot90(image, k=1, axes=(0,1)).copy(),np.rot90(mask, k=1, axes=(0,1)).copy())
    elif choice == 4:
        # add random noise
        noise = np.random.normal(0,1,(1024,1024))
        return(image+noise,mask)
    elif choice == 5:
        # do gausian blur with a 2D gaussian with SD = 1
        return(gaussian_filter(image, sigma=1),mask)
    else:
        return(image,mask)
    



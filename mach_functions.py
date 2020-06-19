import cv2
import matplotlib.pyplot as plt
import numpy as np
import mach_util
import math
import sys
from skimage import transform
import imutils
from scipy.ndimage import measurements

sys.path = ["./matchedmyo/"] + sys.path
import util


def invert_gray_scale(images):
    """
    make cells as white and background as black
    """
    new_images=[]
    for i in images:
        new_images.append(255 - i)
    return new_images


def normalize(image):
    """
    Normolize the gray_scale image
    """
    newimage = image/np.max(image)
    return newimage

def displayMultImage(images,cmap=None,numbering=True):
    """
    Function to display the cropped representative microglia cells
    """
    N = len(images)
    if cmap == None:
        # check the if the image is RGB or grayscale
        if len(np.shape(images[0])) == 2:
                cmap='gray'
        else:
                cmap=None

            
        #now of rows, each row has maxi 25 columns
    nRows = math.ceil(N / 25.0)
    nColumns = 25
        
    figure = plt.figure(figsize=(nColumns,nRows))
            
    for i,img in enumerate(images):
            plt.subplot(nRows,nColumns,i+1)
            plt.imshow(img,cmap=cmap)
            plt.xticks([])
            plt.yticks([])
            if numbering:
                plt.title(str(i+1))
            plt.subplots_adjust(wspace=0,hspace=0)

def PreProcessCroppedCells(oriImage):
    """
    Converts the tif image to gray-scale, then inverse the white/black and finally 
    nornolize it
    """
    
    newimage = normalize(invert_gray_scale(cv2.cvtColor(oriImage,cv2.COLOR_RGB2GRAY)))

    return newimage


def alignRodCellvertically(Image,
                            ImageName,
                            thres = 0.4, #thres value for binary image
                            plotting=False, # plotting the detected rod-direction
                            ):
    """
    automatically align rod-shaped cells vertically
    """
    #image = util.ReadImg(Imagename).astype(np.uint8)
    #grey = cv2.cvtColor(rod,cv2.COLOR_RGB2GRAY)
    dummy, binaryimage = cv2.threshold(Image, thres, 1.0, cv2.THRESH_BINARY)

    # detect the direction of rod using the hough line detection 
    # note, the angle unit is radian (1 radian = 57.2958 degree)
    Hspace, Angle, Dist = transform.hough_line(binaryimage)

    # find the peaks in the hough space
    peaks = transform.hough_line_peaks(Hspace,Angle,Dist)
    if len(peaks[0]) > 1:
        raise RuntimeError('More than 1 directions exist in %s, plz use a image that contains only 1 direction' % (ImageName))


    angle = peaks[1][0]
    dist = peaks[2][0]

    #  now rotate the image based one the angle
    radianTodegree = 57.29

    rotatedImage = imutils.rotate(Image,angle*radianTodegree)
    if plotting:    
        origin = np.array((0, binaryimage.shape[1]))
        line = (dist - origin * np.cos(angle)) / np.sin(angle)
        plt.subplot(1,2,1)
        plt.imshow(image,origin='lower')
        plt.plot(origin,line,'r-')

        plt.ylim([0,binayimage.shape[1]])
        plt.xticks([])
        plt.yticks([])
        plt.title('detected directions')

        plt.subplot(1,2,2)
        plt.imshow(rotatedImage,origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title('align vertically')


    return rotatedImage




def GenerateSubImage(Image,
                    Left = 15,
                    Right = 15,
                    Interval = 5):
    """
    for each aligned image, sample subimages spaning the +15/-15 degree centered at the vertical 
     line, these subimages will be used to generate MACH filter
    """
    rnum = int(Right/Interval)
    lnum = int(Left/Interval)

    angles = []
    for i in np.arange(rnum+1):
        angles.append(i*Interval)
    for i in np.arange(1,lnum+1):
        angles.append(360-i*Interval)

    subimages = []
    for i in angles:
        subimages.append(imutils.rotate(Image,i))

    return subimages
    


def getTrueMarkedCells(annotatedImage,
                        markechannel='red'):
    """
    To extract the positions of true cells from the manually 
    annotated image (by gimp or photoshop).
    The returned object is a binary image with cells as 1 and background as
    0.
    The convolution results of a filter is then compare with this binary image 
    to calculate the True Positive/False Postive.
    """
    rows = annotatedImage.shape[0]
    cols = annotatedImage.shape[1]

    markimage = np.zeros((rows,cols))

    if markechannel == 'red':
        idx = 0
    elif markechannel == "green":
        idx = 1
    else:
        idx = 2
    markimage[annotatedImage[:,:,idx] == 255] = 1.0
        
    return markimage

def calculate_TP(hits,
                    marked_truth,
                    verbose=True,
                    returnOverlap=False):
    '''
    Calculted the True Positive (TP) score
    given the hits (the SNR thresholded binary image)
    and the truth image (binary image)
    '''
    labels, numofgroups = measurements.label(marked_truth)

    overlap = hits*marked_truth

    results = np.zeros(marked_truth.shape)
    for i in np.arange(1,numofgroups+1):
        if np.sum(overlap[labels == i]) > 1:
            results[labels==i] = 1.0

    dummy, numofgroups2 = measurements.label(results)
    TP = numofgroups2/numofgroups
    if verbose == True:
        print ("Found %d cells in the annoated image" % (numofgroups))
        print ("Successfully detected %d cells" % (numofgroups2))
        print ("The True Positive (TP) score is %5.2f" % (numofgroups2/numofgroups))
    
    if returnOverlap:
        return results, TP
    else:
        return TP

def pasteImage(image, thresd_image, color):
    '''
    Set the R/G channel value to 255 to hightlight the
    hits from bad/good filter
    '''
    newimage = image.copy()
    if color == 'green':
        newimage[thresd_image==1,1] = 255
    elif color == 'red':
        newimage[thresd_image==1,0] = 255
    elif color == 'yellow':
        newimage[thresd_image==1,0:2] = 255
    return newimage
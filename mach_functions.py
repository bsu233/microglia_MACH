import cv2
import matplotlib.pyplot as plt
import numpy as np
import mach_util
import math
import sys
from skimage import transform
import imutils
from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage import morphology
import imutils
import yaml

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
    


def calculate_TP(hits,
                marked_truth,
                    verbose=True,
                    FP=False,
                    returnOverlap=False):
    '''
    Calculted the True Positive (TP) score and FP (optional)
    given the hits (the SNR thresholded binary image)
    and the truth image (binary image)
    '''
    labels, numofgroups = measurements.label(marked_truth)

    overlap = hits*marked_truth

    results = np.zeros(marked_truth.shape)
    for i in np.arange(1,numofgroups+1):
        if np.sum(overlap[labels == i]) > 1:
            results[labels==i] = 1.0

    if FP:
        labels3, numofgroups3 = measurements.label(hits)
        overlap3 = hits*results 
        results3 = hits.copy()

        for i in np.arange(1,numofgroups3+1):
            if np.sum(overlap3[labels3 == i]) > 1:
                results3[labels3==i] = 0.0
        
        dumpy1, numberoffp = measurements.label(results3)



    dummy, numofgroups2 = measurements.label(results)
    TP = numofgroups2/numofgroups
    if verbose == True:
        if not FP:
            print ("Found %d cells in the annoated image" % (numofgroups))
            print ("Successfully detected %d cells" % (numofgroups2))
            print ("The TP rate is %5.2f" % (numofgroups2/numofgroups))
        else:
            print ("Found %d cells in the annoated image" % (numofgroups))
            print ("Successfully detected %d cells" % (numofgroups2))
            print ("False positively detected %d cells" % (numberoffp) )
            print ("The TP/FP rate are %5.2f and %5.3f" % (numofgroups2/numofgroups, numberoffp/numgerofgroups))
        
    


    if returnOverlap:
        if FP:
            return results, results3, TP,FP
        else:
            return reultts,TP
    else:
        if FP:
            return TP, FP
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
    elif color == "blue":
        newimage[thresd_image==1,2] = 255
    elif color == "cyan":
        newimage[thresd_image==1,1:3] = 255
    return newimage

def extractTruthMask(annotateImage):
    """
    Return mask (binary images) showing the position of 
    each cell type from the annotateImage
    """
    truthColors = {
    'ram': 'green',
    'rod': 'red',
    'amoe': 'cyan',
    'hyp': 'blue',
    'dys': 'yellow'
    }

    colorchannels = {
    'green' : [1],
    'red' : [0],
    'blue': [2],
    'yellow': [0,1],
    'cyan': [1,2],
    }

    masks = dict()
    for i in truthColors.keys():
        mask = np.zeros((annotateImage.shape[0],annotateImage.shape[1]))
        channel = colorchannels[truthColors[i]]
        if len(channel) == 1:
            loc1 = np.equal(annotateImage[:,:,channel[0]],255)
            colorchannelSum = np.sum(annotateImage,axis=2)
            loc2 = np.equal(colorchannelSum,255)
            loc = np.logical_and(loc1,loc2)
            mask[loc] = 1.0
        else:
            #print (i,channel[0],channel[1])
            loc1 = np.equal(annotateImage[:,:,channel[0]],255)
            loc2 = np.equal(annotateImage[:,:,channel[1]],255)
            
            mask = np.logical_and(loc1,loc2).astype(np.float)
            
        masks[i] = mask
    
    return masks
    
def load_yaml(yamlFile):
    """
    Load parameters defined in the yaml file
    """

    with open(yamlFile) as f:
        data = yaml.load(f)
    return data

def readInfilters(filter_dir,filter_names):
    """
    Read in previously generated MACH filters AS Well as generate a penalty 
    filter for the rod filter

    The output is a dict
    """
    filters = dict()
    
    for i in filter_names:
        filters[i] = np.loadtxt(filter_dir+"/"+i+'_filter') # read in ndarray

    #generate a penalty filter for the rod MACH filter
    norm_rod = filters['rod']/np.max(filters['rod'])
    mask = np.greater(norm_rod,0.4).astype(np.float)
    mask = morphology.binary_dilation(mask,iterations=2)
    filters['rodp'] = filters['ram'].copy()
    filters['rodp'][mask == 1.0] = 0.0

    np.savetxt(filter_dir+'rodp_filter',filters['rodp'])

    return filters

def RodPenalty(rod_hits,rodp_hits,\
                rodthres=0.32,\
                rodpthres=0.18,
                ):
    """
    Retrun a binary image representing the results of
    "rod_hits - penalty_hits"
    where rod_hits and penaly_hits are convolution results of image against the
    rod filter and rodp (penanlty) filter 
    """
    rod_mask = np.greater(rod_hits,rodthres).astype(np.float)
    penaltymask = np.greater(rodp_hits,rodpthres).astype(np.float)
    penaltymask = morphology.binary_closing(penaltymask,iterations=3)
    final_results = rod_mask.copy()

    a1, a2 = measurements.label(rod_mask)
    for i in np.arange(a2):
        if np.sum(final_results[a1 == i+1]) <= 75*75*0.06: #area restraint
            final_results[a1 == i+1] = 0
        if  np.sum(penaltymask[a1 == i+1]) != 0:
            final_results[a1 == i+1] = 0
    
    return final_results # a binary image

def Rodrotate(TestImage,rodfilter,rodpfilter,\
            rod_thres = 0.3,\
            rodp_thres = 0.2,\
            iters=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]):
    """
    Rotate the rodfilter and rodp filter with angles in the iters list,
    For each angle, convolve the TestImage against the rod filter and rodp filter

    The final output is a binary image showing the detection of cells when putting all angles
    together (logical_or)
    """
    
    final_results = np.zeros((TestImage.shape[0],TestImage.shape[1]))
    for i in iters:
        rotated_rod = imutils.rotate(rodfilter,i)
        rotated_rodp = imutils.rotate(rodpfilter,i)
        rod_hits = ndimage.convolve(TestImage,rotated_rod)
        rodp_hits = ndimage.convolve(TestImage,rotated_rodp)
        results = RodPenalty(rod_hits,rodp_hits,rodthres= rod_thres,rodpthres = rodp_thres)

        # megre the results of this angle into the final_results
        final_results = np.logical_or(final_results,results).astype(np.float)

    return final_results



    


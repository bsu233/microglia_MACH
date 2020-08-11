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
import os

sys.path = ["./matchedmyo/"] + sys.path
import util


truthColors = {
    'ram': 'green',
    'rod': 'red',
    'amoe': 'cyan',
    'hyp': 'blue',
    'dys': 'yellow'
    }


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
        plt.imshow(Image,origin='lower')
        plt.plot(origin,line,'r-')

        plt.ylim([0,binaryimage.shape[1]])
        plt.xticks([])
        plt.yticks([])
        plt.title('detected directions')

        plt.subplot(1,2,2)
        plt.imshow(rotatedImage,origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title('align vertically')


    return rotatedImage

def construct_test_image(filternames,perct=None, nRows=50,nCols=50):
    '''
    construct a test image (size = nRows x nCols filters) using filternames (a list containts the name)
    and the percentage of each filter
    TODO: not working, need to change filternames to image list to make it general
    '''
    filter_dimes = np.shape(filters[filternames[0]])
    image_size_x = filter_dimes[0]*nRows
    image_size_y = filter_dimes[1]*nCols
    
    test_image = np.ones((image_size_x,image_size_y))
    
    
    numFilters = len(filternames) # the number of filters
    
    for m in np.arange(nRows):
        for n in np.arange(nCols):
            prob = np.random.rand()
            for j in np.arange(numFilters):
                    if j == 0:
                        left = 0
                    else: 
                        left = np.sum(perct[0:j])
                    right = left + perct[j]
                    if prob >= left and prob <= right:
                        indx = j

            test_image[m*filter_dimes[0]:(m+1)*filter_dimes[0],\
                        n*filter_dimes[1]:(n+1)*filter_dimes[1]] = filters[filternames[indx]]
            
    # now add some gaussian noise
    test_image = add_gaussian_noise(test_image,var=0.003)
       
    return test_image



def azimuthally_psd(image,bins=30):
    '''
    generate the PSD along radial direction, the image is grayscale
    '''
    fftimage = np.fft.fft2(image)
    
    dimes = fftimage.shape
    centerX = int(dimes[0]/2)
    centerY = int(dimes[1]/2)
    maskimage = np.ones(dimes)
    maskimage[centerX,centerY] = 0
    
    edt_mask = ndimage.distance_transform_edt(maskimage)
    
    ori_psd = np.log(np.abs(np.fft.fftshift(fftimage))**2)
    # max radial length
    mlen = np.sqrt(centerX**2+centerY**2)
    spacing = mlen/bins
    
    
    psd = []
    dist = []
    for i in np.arange(bins):
        low_limit = i*spacing
        high_limit = (i+1)*spacing
        dist_i = 0.5*(low_limit + high_limit)
        
        indx = np.logical_and(np.greater_equal(edt_mask,low_limit),np.less(edt_mask,high_limit))
        psd_i = np.sum(ori_psd[indx])/np.sum(maskimage[indx])
        psd.append(psd_i)
        dist.append(dist_i)
        
    return dist,psd



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
    


def divde_big_image(bigimage,grayscale=True):
    """
    divided the big grayscale image into 
    100 small images
    """
    dx = int(bigimage.shape[0]/10)
    dy = int(bigimage.shape[1]/10)
    
    if grayscale:
        small_images = [bigimage[dx*i:dx*(i+1),dy*j:dy*(j+1)]\
                for i in np.arange(10) for j in np.arange(10)]
    else:
        small_images = [bigimage[dx*i:dx*(i+1),dy*j:dy*(j+1),:]\
                for i in np.arange(10) for j in np.arange(10)]
    return small_images


def getTPFP(detectedCells,
                marked_truth,
                verbose=True,
                returnOverlap=False):
    '''
    Calculted the True Positive (TP) score and False Positive (FP) score
    given the detected cells by MACH protocol and the annotated truth image
    Both the detected cells and marked_truth are binary images.
    '''
    labels, numofgroups = measurements.label(marked_truth)

    overlap = detectedCells*marked_truth

    results = np.zeros(marked_truth.shape)
    for i in np.arange(1,numofgroups+1):
        if np.sum(overlap[labels == i]) > 1:
            results[labels==i] = 1.0

    
    labels3, numofgroups3 = measurements.label(detectedCells)
    overlap3 = detectedCells*results 
    results3 = detectedCells.copy()

    for i in np.arange(1,numofgroups3+1):
        if np.sum(overlap3[labels3 == i]) > 1:
            results3[labels3==i] = 0.0
        
    dummy1, numberoffp = measurements.label(results3)



    dummy2, numofgroups2 = measurements.label(results)
    TP = numofgroups2/numofgroups
    FP = numberoffp/numofgroups
    if verbose == True:
        print ("Found %d cells in the annoated image" % (numofgroups))
        print ("Successfully detected %d cells" % (numofgroups2))
        print ("False positively detected %d cells" % (numberoffp) )
        print ("The TP/FP rate are %5.2f and %5.3f" % (numofgroups2/numofgroups, numberoffp/numofgroups))
    
    # store the cell numbers 
    cellNums = dict()
    cellNums['AnnotatedCell'] = numofgroups
    cellNums['TPCells'] = numofgroups2
    cellNums['FPCells'] = numberoffp
        

    if returnOverlap:
        return results, results3, TP,FP, cellNums
    else:
        return TP, FP, cellNums
    

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

def addGrid(image,correlationPlane=False,spacing=75,lw=2):
    """
    Add grid onto the image to help visulize the resutls
    The correlation plane (covolution results) is a special case as we
    need to keep the vmax/vmin of the colorbar unchanged
    """
    newImage=image.copy()
    X = image.shape[0]
    Y = image.shape[1]
    xgrid = np.arange(0,X,spacing)
    ygrid = np.arange(0,Y,spacing)
    
    if len(image.shape) == 3:
        vmax = 255
    else:
        vmax = 1.0

    if correlationPlane:
        vmax = (np.min(image) + (np.max(image)-np.min(image))*0.5)/0.3

    for i in xgrid:
        newImage[i-lw:i+lw,:] = vmax*0.3
    for j in ygrid:
        newImage[:,j-lw:j+lw] = vmax*0.3

    return newImage 


def extractTruthMask(annotateImage):
    """
    Return mask (binary images) showing the position of 
    each cell type from the annotateImage
    """
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

def readInfilters(Params):
    """
    Read in previously generated MACH filters 
    The output is a dict
    """
    filter_dir = Params['FilterDir']
    
    filters = dict()
    for i in Params['FilterNames'].keys():
        filters[i] = dict()
        for j in Params['FilterNames'][i]:
            filter = np.loadtxt(filter_dir+"/"+j+'_filter') # read in ndarray
        
            filters[i][j] = filter
    
    return filters

def readInImages(Params):
    """
    Read in the images that will be used for detection
    If it is validation, also read in the annotated true image
    """
    Images = dict()

    if not os.path.exists(Params['InputImage']):
        raise RuntimeError(f"The input image {Params['InputImage']} does not exist, plz double check the image path")

    image = cv2.imread(Params['InputImage'])
    colorImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gray_scale_image = PreProcessCroppedCells(colorImage)
    Images['colorImage'] = colorImage
    Images['grayImage'] = gray_scale_image

    if Params['TrueImage'] != None:
    
        if not  os.path.exists(Params['TrueImage']):
            raise RuntimeError(f"The annotated image {Params['TrueImage']} does not exist, plz double check the image path")

        truthImage = cv2.imread(Params['TrueImage'])
        truthImage = cv2.cvtColor(truthImage,cv2.COLOR_BGR2RGB)
        Images['trueImage'] = truthImage
        masks = extractTruthMask(truthImage)
        Images['masks'] = masks
    else:
        Images['trueImage'] = None
        Images['masks'] = None
    
    return Images
 
def RodSNR(image,filter,penaltyfilter,snrthres,returnCorrelationPlane=False):
    """
    Using the rod filter and a penalty filter to detected the 
    rod-shaped cells
    SNF = Cr/Crp where Cr is the convolution results of rod filter and
    Crp is the convolution results of the rod penalty filter
    """
    Cr = ndimage.convolve(image,filter)
    Crp = ndimage.convolve(image,penaltyfilter)
    
    SNR = Cr/Crp

    # thresholding SNR
    thresSNR = np.greater(SNR,snrthres).astype(np.float)
    
    if returnCorrelationPlane:
        return thresSNR, Cr, Crp
    else:
        return thresSNR


#### Temporarily deprecated
#def RodPenalty(rod_hits,rodp_hits,\
#                rodthres=0.32,\
#                rodpthres=0.18,
#                ):
#    """
#    Retrun a binary image representing the results of
#    "rod_hits - penalty_hits"
#    where rod_hits and penaly_hits are convolution results of image against the
#    rod filter and rodp (penanlty) filter 
#    """
#    rod_mask = np.greater(rod_hits,rodthres).astype(np.float)
#    penaltymask = np.greater(rodp_hits,rodpthres).astype(np.float)
#    penaltymask = morphology.binary_closing(penaltymask,iterations=3)
#    final_results = rod_mask.copy()
#
#    a1, a2 = measurements.label(rod_mask)
#    for i in np.arange(a2):
#        if np.sum(final_results[a1 == i+1]) <= 75*75*0.06: #area restraint
#            final_results[a1 == i+1] = 0
#        if  np.sum(penaltymask[a1 == i+1]) != 0:
#
#            final_results[a1 == i+1] = 0
#    
#   return final_results # a binary image


#def Rodrotate(TestImage,rodfilter,rodpfilter,\
#            rod_thres = 0.3,\
#            rodp_thres = 0.2,\
#            iters=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]):
#    """
#    Rotate the rodfilter and rodp filter with angles in the iters list,
#    For each angle, convolve the TestImage against the rod filter and rodp filter
#
#    The final output is a binary image showing the detection of cells when putting all angles
#    together (logical_or)
#    """
    
#    final_results = np.zeros((TestImage.shape[0],TestImage.shape[1]))
#    for i in iters:
#        rotated_rod = imutils.rotate(rodfilter,i)
#        rotated_rodp = imutils.rotate(rodpfilter,i)
#        rod_hits = ndimage.convolve(TestImage,rotated_rod)
#        rodp_hits = ndimage.convolve(TestImage,rotated_rodp)
#        results = RodPenalty(rod_hits,rodp_hits,rodthres= rod_thres,rodpthres = rodp_thres)

#        # megre the results of this angle into the final_results
#        final_results = np.logical_or(final_results,results).astype(np.float)

#    return final_results
#
def giveRod(TestImage,rodfilter,penaltyfilter,snrthres,iters=[20,40,60,80,100,120,140,160,180]):
    """
    Rotate the rod and pennalty filter for each angle in the iters,
    get the SNR and the thresSNR results, merge the detected cells at each angle and give the 
    final deteced rod cells
    """
     CrPlanes = dict()

     final_results = np.zeros((TestImage.shape[0],TestImage.shape[1]))
     for i in iters:
        rotated_rod = imutils.rotate(rodfilter,i)
        rotated_rodp = imutils.rotate(penaltyfilter,i)
        rodcells, Cr, Crp  = RodSNR(TestImage,rotated_rod,rotated_rodp,snrthres=snrthres,returnCorrelationPlane=True)
    # megre the results of this angle into the final_results
        final_results = np.logical_or(final_results,results).astype(np.float)
        CrPlanes[i] = [Cr,Crp]

    return final_results, CrPlanes

    

def giveRamHyp(image,ramp_filter,hypp_filter,rampthres=0.22,hyppthres=0.18):
    """
    give the detected ramified cells and hypertrophic cells 
    based on the hits of ramifiled process filter (ramp) and
    hyertrophic process filter (hypp)
    """

    ramp_hits = ndimage.convolve(image,ramp_filter)
    hypp_hits = ndimage.convolve(image,hypp_filter)
    ramp_detected = np.greater(ramp_hits,rampthres).astype(np.float)
    hypp_detected = np.greater(hypp_hits,hyppthres).astype(np.float)

    # the overlapped area between ramp_detected and hypp_detected are assigned as hyptrophic cells
    # the remaining part in the ramp_detected are ramified cells


    labels, numofgroups = measurements.label(ramp_detected)

    overlap = ramp_detected*hypp_detected

    results = np.ones(ramp_detected.shape)
    for i in np.arange(1,numofgroups+1):
        if np.sum(overlap[labels == i]) != 0:
            results[labels==i] = 0.0

    ram_cells = ramp_detected.copy()
    hyp_cells = np.zeros(ramp_detected.shape)

    ram_cells[results==0] = 0

    hyp_cells[results == 0] = 1
    
    
    return ram_cells, hyp_cells

def giveAmoeDys(image,amoefilter,amoethres=0.22,areathes=20):
    """
    give the detected amoeboid and dystrophic cells 
    based on the convolution hits of amoeboid filter 

    since the amoe and dys filters ahve similar convolution results
    we only use the amoe_hits
    First thres the aomoe hits and discrimnate amoe/dys cells 
    based on area (dys > amoe)
    """
    amoe_hits = ndimage.convolve(image,amoefilter)
    amoe_detected = np.greater(amoe_hits,amoethres).astype(np.float)



    labels, numofgroups = measurements.label(amoe_detected)
    amoe_cells = amoe_detected.copy()
    dys_cells = amoe_detected.copy()

    results = np.ones(amoe_detected.shape)
    for i in np.arange(1,numofgroups+1):
        
        if np.sum(amoe_detected[labels == i]) <= areathes:
            dys_cells[labels == i] = 0.0
        else:
            amoe_cells[labels == i] = 0.0

    return amoe_cells, dys_cells

def removeDetectedCells(TestImage,detected_cells,bgmthres=0.4):
    """
    Remove the detected cells of one filter from the test image so that we can 
    apply the next filter

    TestImage: gray-scale image
    detected_cells: binary image of detected cells using one filter
    bgmthres: thresholding value for cells/background 

    output: new TestImage
    """
    # generate a binary image of the test image
    bimage = np.greater(TestImage,bgmthres).astype(np.float)

    #background gray-scale value
    bgmValue = np.average(TestImage[bimage==0])
    labels, numofgroups = measurements.label(bimage)

    overlap = detected_cells*bimage

    results = np.ones(bimage.shape)
    for i in np.arange(1,numofgroups+1):
        if np.sum(overlap[labels == i]) != 0:
            results[labels==i] = 0.0

    newtestImage = TestImage.copy()
    newtestImage[results==0] = bgmValue
    
    return newtestImage

    
class machPerformance:
    """
    A class to store the peformance of mach filtering
    """

    def __init__(self, mach_results, masks):
        """
        The mach_results a dict that contains the detected_cells by the filtering 
        protocol
        And the annotated masks for each type of cells
        """
        self.detectedCells = mach_results
        self.masks = masks
        self.CellNums = dict()
        self.TP = dict()
        self.TP_weights = dict()
        self.FP = dict()
        self.FP_weights = dict()
        self.TP_cells = dict()
        self.FP_cells = dict()
        self.PFscore = 0.0

    def resetTPFPWeights(self):
        """
        setting the weights of TP/FP to 1.0
        """
        for cellType in self.detectedCells.keys():
            self.TP_weights[cellType] = 1.0
            self.FP_weights[cellType] = 1.0

    def setTPWeights(self,newTPWeights):

        for cType, newW in newTPWeights.items():
            self.TP_weights[cType] = newW
    
    def setFPWeights(self,newFPWeights):
        for cType, newW in newFPWeights.items():
            self.FP_weights[cType] = newW    
        

    def calculateTPFP(self):
        for cellType, dCells in self.detectedCells.items():
            print (cellType)
            TP_cells, FP_cells, TP, FP, cellNums = getTPFP(dCells,self.masks[cellType],verbose=False,returnOverlap=True)
            self.TP[cellType] = TP
            self.FP[cellType] = FP
            self.TP_cells[cellType] = TP_cells
            self.FP_cells[cellType] = FP_cells
            self.CellNums[cellType] = cellNums
    

    def calculatePFscore(self):
        """
        Calcualte the Performance Criterion based on the TP/FP score
        of each cell type and the associated weights.
        """
        PFscore = 0
        for i in self.detectedCells.keys():
            PFscore += self.TP_weights[i]*(1-self.TP[i]) + self.FP_weights[i]*self.FP[i]

        self.PFscore = PFscore
        return PFscore

    def plotTPFPfigure(self,figname):
        """
        Plot the TP/FP score of each cell type
        """
        annotatedCellNums = dict()
        TPCellNums = dict()
        FPCellNums = dict()

        for cellType in self.detectedCells.keys():
            annotatedCellNums[cellType] = self.CellNums[cellType]['AnnotatedCell']
            TPCellNums[cellType] = self.CellNums[cellType]['TPCells']
            FPCellNums[cellType] = self.CellNums[cellType]['FPCells']

        width=0.4
        barlist1 = plt.bar(np.arange(len(annotatedCellNums.keys())),annotatedCellNums.values(),width=width,alpha=0.3)
        barlist2 = plt.bar(np.arange(len(TPCellNums.keys())),TPCellNums.values(),width=width,alpha=1)
        barlist3 = plt.bar(np.arange(len(FPCellNums.keys()))+width,FPCellNums.values(),width=width,alpha=0.3,hatch="*",ecolor='b')

        for i, j in enumerate(barlist1):
            j.set_color(truthColors[list(annotatedCellNums.keys())[i]])
        for i, j in enumerate(barlist2):
            j.set_color(truthColors[list(TPCellNums.keys())[i]])
        for i, j in enumerate(barlist3):
            j.set_color(truthColors[list(FPCellNums.keys())[i]])

        plt.ylabel('Cell Nums')
        plt.xticks(np.arange(len(annotatedCellNums.keys())),list(annotatedCellNums.keys()))
        lg = plt.legend([barlist1[0],barlist2[0],barlist3[0]],['Annotated Cells','TP Cells','FP Cells'],loc=(1, 0.75))
        plt.savefig(figname,bbox_extra_artists=(lg,), 
            bbox_inches='tight',
            dpi=300)
    

            

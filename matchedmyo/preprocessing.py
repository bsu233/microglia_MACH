from __future__ import print_function
###
### Group of functions that will walk the user fully through the preprocessing
### routines.
###
import sys
import os
import cv2
import util
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pygame, sys
from PIL import Image
pygame.init() # initializes pygame modules
from sklearn.decomposition import PCA
import imutils
import matchedFilter as mF
import tifffile

### Global variable to determine if the image is too large for certain operations (such as rotation, etc.)
img_size_cutoff = 36e6 # [total pixels]

###############################################################################
###
### Normalization Routines
###
##############################################################################

def normalizeToStriations(img, subsectionIdxs,filterSize):
  '''
  function that will go through the subsection and find average smoothed peak 
  and valley intensity of each striation and will normalize the image 
  based on those values.
  '''

  print ("Normalizing myocyte to striations")
  
  ### Load in filter that will be used to smooth the subsection
  thisPath = os.path.realpath(__file__).split('/')[:-1]
  thisPath = '/'.join(thisPath)
  WTfilterName = thisPath+'/myoimages/singleTTFilter.png'
  WTfilter = util.LoadFilter(WTfilterName)

  if img.dtype != np.uint8:
    img = img.astype(np.float32) / float(np.max(img)) * 255.
    img = img.astype(np.uint8)

  ### Perform smoothing on subsection
  smoothed = np.asarray(mF.matchedFilter(img,WTfilter,demean=False))

  ### Grab subsection of the smoothed image
  smoothedSubsection = smoothed.copy()[subsectionIdxs[0]:subsectionIdxs[1],
                                       subsectionIdxs[2]:subsectionIdxs[3]]

  ### Now we have to normalize to 255 for cv2 algorithm to work
  if img.dtype != np.uint8:
    smoothedSubsection = smoothedSubsection * 255. / np.max(smoothedSubsection)
    smoothedSubsection = smoothedSubsection.astype(np.uint8)
  #plt.figure()
  #plt.imshow(smoothedSubsection)
  #plt.colorbar()
  #plt.show()
  
  ### Perform Gaussian thresholding to pull out striations
  # blockSize is pixel neighborhood that each pixel is compared to
  blockSize = int(round(float(filterSize) / 3.57)) # value is empirical
  # blockSize must be odd so we have to check this
  if blockSize % 2 == 0:
    blockSize += 1
  # constant is a constant that is subtracted from each distribution for each pixel
  constant = 0
  # threshValue is the value at which super threshold pixels are marked, else px = 0
  threshValue = 1
  gaussSubsection = cv2.adaptiveThreshold(smoothedSubsection, threshValue,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, blockSize,
                                          constant)
  #plt.figure()
  #plt.imshow(gaussSubsection)
  #plt.colorbar()
  #plt.show()

  ### Calculate the peak and valley values from the segmented image
  peaks = smoothedSubsection[np.nonzero(gaussSubsection)]
  peakValue = np.mean(peaks)
  peakSTD = np.std(peaks)
  valleys = smoothedSubsection[np.where(gaussSubsection == 0)]
  valleyValue = np.mean(valleys)
  valleySTD = np.std(valleys)

  print ("Average Striation Value:", peakValue)
  print ("Standard Deviation of Striation:", peakSTD)
  print ("Average Striation Gap Value:", valleyValue)
  print ("Stand Deviation of Striation Gap", valleySTD)

  ### Calculate ceiling and floor thresholds empirically
  ceiling = peakValue + 3 * peakSTD
  floor = valleyValue - valleySTD
  if ceiling > 255:
    ceiling = 255.
  if floor < 0:
    floor = 0
  
  ceiling = int(round(ceiling))
  floor = int(round(floor))
  print ("Ceiling Pixel Value:", ceiling)
  print ("Floor Pixel Value:", floor)

  ### Threshold
  #img = img.astype(np.float64)
  #img /= np.max(img)  
  img[img>=ceiling] = ceiling
  img[img<=floor] = floor
  img -= floor
  img = img.astype(np.float64)
  img /= np.max(img)
  img *= 255
  img = img.astype(np.uint8)

  return img

def normalizeToStriations_given_subsection(img,subsection,filterSize):
  '''Does the same thing as "normalizeToStriations" but instead of performing matched filtering 
  with the whole image, just does it with the subsection to avoid long computation times.'''
  print ("Normalizing myocyte to striations")
  
  ### Load in filter that will be used to smooth the subsection
  thisPath = os.path.realpath(__file__).split('/')[:-1]
  thisPath = '/'.join(thisPath)
  WTfilterName = thisPath+'/myoimages/singleTTFilter.png'
  WTfilter = util.LoadFilter(WTfilterName)

  if subsection.dtype != np.uint8:
    subsection = subsection.astype(np.float32) / float(np.max(subsection)) * 255.
    subsection = subsection.astype(np.uint8)
  if img.dtype != np.uint8:
    img = img.astype(np.float32) / float(np.max(img)) * 255.
    img = img.astype(np.uint8)

  ### Perform smoothing on subsection
  smoothedSubsection = np.asarray(mF.matchedFilter(subsection,WTfilter,demean=False))

  ### Now we have to normalize to 255 for cv2 algorithm to work
  if smoothedSubsection.dtype != np.uint8:
    # smoothedSubsection = smoothedSubsection * 255. / np.max(smoothedSubsection)
    # smoothedSubsection = smoothedSubsection.astype(np.uint8)
    smoothedSubsection = smoothedSubsection.astype(np.uint8) # should already be normalized to 255
  
  ### Perform Gaussian thresholding to pull out striations
  # blockSize is pixel neighborhood that each pixel is compared to
  blockSize = int(round(float(filterSize) / 3.57)) # value is empirical
  # blockSize must be odd so we have to check this
  if blockSize % 2 == 0:
    blockSize += 1
  # constant is a constant that is subtracted from each distribution for each pixel
  constant = 0
  # threshValue is the value at which super threshold pixels are marked, else px = 0
  threshValue = 1
  gaussSubsection = cv2.adaptiveThreshold(smoothedSubsection, threshValue,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, blockSize,
                                          constant)

  ### Calculate the peak and valley values from the segmented image
  peaks = smoothedSubsection[np.nonzero(gaussSubsection)]
  peakValue = np.mean(peaks)
  peakSTD = np.std(peaks)
  valleys = smoothedSubsection[np.where(gaussSubsection == 0)]
  valleyValue = np.mean(valleys)
  valleySTD = np.std(valleys)

  print ("Average Striation Value:", peakValue)
  print ("Standard Deviation of Striation:", peakSTD)
  print ("Average Striation Gap Value:", valleyValue)
  print ("Stand Deviation of Striation Gap", valleySTD)

  ### Calculate ceiling and floor thresholds empirically
  ceiling = peakValue + 3 * peakSTD
  floor = valleyValue - valleySTD
  if ceiling > 255:
    ceiling = 255.
  if floor < 0:
    floor = 0
  
  ceiling = int(round(ceiling))
  floor = int(round(floor))
  print ("Ceiling Pixel Value:", ceiling)
  print ("Floor Pixel Value:", floor)

  ### Threshold 
  img[img>=ceiling] = ceiling
  img[img<=floor] = floor
  img -= floor
  img = img.astype(np.float64)
  img /= np.max(img)
  img *= 255
  img = img.astype(np.uint8)

  return img

###############################################################################
###
### FFT Filtering Routines
###
###############################################################################

# clearly I've done a lot of work here

###############################################################################
###
###  Reorientation Routines
###
###############################################################################

def autoReorient(img):
  '''
  Function to automatically reorient the given image based on principle component
    analysis. This isn't incorporated into the full, 'robust' preprocessing routine
    but is instead used for preprocessing the webserver uploaded images since 
    we want as little user involvement as possible.

  INPUTS:
    - img: The image that is to be reoriented. Uploaded as a float with 0 <= px <= 1.
  '''
  raise RuntimeError( "Broken for some reason. Come back to debug")

  dummy = img.copy()
  dumDims = np.shape(dummy)
  minDim = np.min(dumDims)
  maxDim = np.max(dumDims)
  argMaxDim = np.argmax(dumDims)
  diff = maxDim - minDim
  padding = int(round(diff / 2.))
  if argMaxDim == 1:
    padded = np.zeros((minDim+2*padding,maxDim))
    padded[padding:-padding,:] = dummy
  else:
    padded = np.zeros((maxDim,minDim+2*padding))
    padded[:,padding:-padding] = dummy
  plt.figure()
  plt.imshow(padded)
  plt.show()
  quit()
  

  pca = PCA(n_components=2)
  pca.fit(padded)
  majorAxDirection = pca.explained_variance_
  yAx = np.array([0,1])
  degreeOffCenter = (180./np.pi) * np.arccos(np.dot(yAx,majorAxDirection)\
                    / (np.linalg.norm(majorAxDirection)))

  print ("Image is", degreeOffCenter," degrees off center")

  ### convert img to cv2 acceptable format
  acceptableImg = np.asarray(img * 255.,dtype=np.uint8)
  rotated = imutils.rotate_bound(acceptableImg, -degreeOffCenter)

  return rotated

def setup(array):
    #px = pygame.image.load(path)
    # using a try/except here to catch images that are too large to display
    try:
      px = pygame.surfarray.make_surface(array)
      idx_adjustment = (0,0)
    except:
      mid_x = array.shape[0] // 2; mid_y = array.shape[1] // 2
      idx_adjustment = (mid_x - 1000, mid_y - 1000)
      px = pygame.surfarray.make_surface(array[mid_x - 1000:mid_x + 1000, mid_y - 1000:mid_y + 1000])
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px, idx_adjustment

def displayImageLine(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    xNew = pygame.mouse.get_pos()[0]
    yNew = pygame.mouse.get_pos()[1]
    width = xNew - x
    height = yNew - y
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current
    
    # draw line on the image
    red = (255, 0, 0)
    startPoint = topleft
    endPoint = (xNew,yNew)
    screen.blit(px,px.get_rect())
    pygame.draw.line(screen,red,startPoint,endPoint)
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoopLine(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImageLine(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsectionLine(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px, idx_adjustment = setup(newArray)
    pygame.display.set_caption("Draw a Line Orthogonal to Transverse Tubules")
    left, upper, right, lower = mainLoopLine(screen, px)
    pygame.display.quit()
    
    directionVector = (right-left,upper-lower)
    return directionVector

def reorient(img):
  '''Function to reorient the myocyte based on a user selected line that is
  orthogonal to the TTs'''

  print ("Reorienting Myocyte")

  ### get direction vector from line drawn by user
  dVect = giveSubsectionLine(img)

  ### we want rotation < 90 deg so we ensure correct axis
  if dVect[0] >= 0:
    xAx = [1,0]
  else:
    xAx = [-1,0]

  ### compute degrees off center from the direction vector
  dOffCenter = (180./np.pi) * np.arccos(np.dot(xAx,dVect)/np.linalg.norm(dVect))

  ### ensure directionality is correct 
  if dVect[1] <= 0:
    dOffCenter *= -1
  print ("Image is",dOffCenter,"degrees off center")

  ### rotate image
  if np.prod(img.shape) < img_size_cutoff:
    rotated = imutils.rotate_bound(img,dOffCenter)
  else:
    print ("Image is too large to rotate. Instead the filter rotation angles will be",
           "adjusted accordingly. Developers: Keep in mind this invalidates the indexes", 
           "returned by these preprocessing routines.")
    rotated = img

  return rotated, dOffCenter

###############################################################################
###
### Resizing Routines
###
###############################################################################

def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsection(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px, idx_adjustment = setup(newArray)
    pygame.display.set_caption("Draw a Rectangle Around Several Conserved Transverse Tubule Striations")
    left, upper, right, lower = mainLoop(screen, px)

    # apply index adjustments if necessary
    upper += idx_adjustment[0]; lower += idx_adjustment[0]
    left += idx_adjustment[1]; right += idx_adjustment[1]

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    subsection = array.copy()[upper:lower,left:right]
    indexes = np.asarray([upper, lower, left, right])
    subsection = np.asarray(subsection, dtype=np.float64)
    pygame.display.quit()
    return subsection, indexes

def resizeToFilterSize(img,filterTwoSarcomereSize,dOffCenter=None):
  '''  Function to semi-automate the resizing of the image based on the filter'''

  print ("Resizing myocyte based on user selected subsection")

  ### 1. Select subsection of image that exhibits highly conserved network of TTs
  if np.prod(img.shape) < img_size_cutoff:
    subsection,indexes = giveSubsection(img)#,dtype=np.float32)
  else:
    # image is too big so we have to use a subsection rotated by the previously informed offset angle
    mid_row = img.shape[0] // 2; mid_col = img.shape[1] // 2
    smaller_img = img[
      mid_row - 1000:mid_row + 1000,
      mid_col - 1000:mid_col + 1000
    ]
    smaller_img = imutils.rotate_bound(smaller_img,dOffCenter)
    # now we can grab a portion of the smaller_img to resize based on TT information
    subsection,indexes = giveSubsection(smaller_img)

  # best to normalize the subsection for display purposes
  subsection /= np.max(subsection)

  ### 2. Resize based on the subsection
  resized, scale, newIndexes = resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes)

  print ("Image Resizing Scale:",scale)
  
  return resized,scale,subsection,newIndexes

def resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes):
  '''
  Function to resize img given a subsection of the image
  '''
  ### Using this subsection, calculate the periodogram
  fBig, psd_Big = signal.periodogram(subsection)
  # finding sum, will be easier to identify striation length with singular dimensionality
  bigSum = np.sum(psd_Big,axis=0)

  ### Mask out the noise in the subsection periodogram
  # NOTE: These are imposed assumptions on the resizing routine
  maxStriationSize = 50.
  minStriationSize = 5.
  minPeriodogramValue = 1. / maxStriationSize
  maxPeriodogramValue = 1. / minStriationSize
  bigSum[fBig < minPeriodogramValue] = 0.
  bigSum[fBig > maxPeriodogramValue] = 0.

  display = False
  if display:
    plt.figure()
    plt.plot(fBig,bigSum)
    plt.title("Collapsed Periodogram of Subsection")
    plt.show()

  ### Find peak value of periodogram and calculate striation size
  striationSize = 1. / fBig[np.argmax(bigSum)]
  imgTwoSarcomereSize = int(round(2 * striationSize))
  print ("Two Sarcomere size:", imgTwoSarcomereSize,"Pixels per Two Sarcomeres")

  if imgTwoSarcomereSize > 70 or imgTwoSarcomereSize < 10:
    print ("WARNING: Image likely failed to be properly resized. Manual resizing",\
           "may be necessary!!!!!")

  ### Using peak value, resize the image
  scale = float(filterTwoSarcomereSize) / float(imgTwoSarcomereSize)
  resized = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  ### Find new indexes in image
  newIndexes = indexes * scale
  newIndexes = np.round(newIndexes).astype(np.int32)

  return resized, scale, newIndexes


###############################################################################
###
### CLAHE Routines
###
###############################################################################

def applyCLAHE(img,filterTwoSarcomereSize):
  print ("Applying CLAHE to Myocyte")

  if img.dtype != np.uint8:
    # convert to uint8 data type to use with clahe algorithm
    fixed = True
    oldImgMax = np.max(img)
    img = img * 255. / float(np.max(img))
    img = img.astype(np.uint8)

  kernel = (filterTwoSarcomereSize, filterTwoSarcomereSize)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=kernel)

  clahedImage = clahe.apply(img)

  if fixed:
    img = img.astype(np.float32) / float(np.max(img)) * oldImgMax

  return clahedImage
  
###############################################################################
###
### Main Routines
###
###############################################################################

def preprocess(fileName, filterTwoSarcomereSize, maskImg=None, writeImage=False, inputs=None):
  '''The routine that handles all of the preprocessing subroutines. This is the routine that we'll
  call from our main script.'''

  img = util.ReadImg(fileName)

  ### Check if the image is too large to be displayed fully using the current GUI
  if np.prod(img.shape) > img_size_cutoff: img_too_large = True
  else: img_too_large = False
  
  ### Make an assumption that if the image is too large, we'll likely need to CLAHE the whole thing
  ### due to dye imbalances across the image.
  if img_too_large: 
    ## Convert to acceptable cv2 format
    prev_img_max = np.max(img)
    prev_img_min = np.min(img)
    prev_img_dtype = img.dtype
    img = (img - prev_img_min) / (prev_img_max - prev_img_min) * 255.
    img = img.astype(np.uint8)
    ## smooth image 
    img = cv2.blur(img,(3,3))
    ## clahe the image
    img = util.ApplyCLAHE(
      img,
      tuple([int(1./8. * dim) for dim in np.shape(img)])
    )
    ## convert back to previous format
    img = img.astype(np.float32) / float(np.max(img)) 
    img *= (prev_img_max - prev_img_min)
    img += prev_img_min

  ### Reorient the image
  img,degreesOffCenter = reorient(img)

  ## If the image is too large, we have to modify the filter rotations instead of rotating the image
  if img_too_large: inputs.dic['iters'] = [it - degreesOffCenter for it in inputs.dic['iters']]

  img,resizeScale,subsection,idxs = resizeToFilterSize(img,filterTwoSarcomereSize,dOffCenter=degreesOffCenter)
  img = applyCLAHE(img,filterTwoSarcomereSize)
  if not img_too_large:
    img = normalizeToStriations(img,idxs,filterTwoSarcomereSize)
  else:
    img = normalizeToStriations_given_subsection(img,subsection,filterTwoSarcomereSize)
    writeImage = True # so we can overlay the results onto the resize and preprocessed image later on.

  # fix mask based on img orientation and resize scale
  if maskImg is not None:
    # if inputs is not None:
    #   inputs.maskImg = processMask(degreesOffCenter,resizeScale,fileName=fileName, maskImg = maskImg, imgShape = img.shape)
    # else:
    maskImg = processMask(degreesOffCenter,resizeScale,fileName=fileName, maskImg = maskImg, imgShape = img.shape)

  # write file
  if writeImage or img_too_large:
    name,fileType = fileName[:-4],fileName[-4:]
    newName = name+"_processed"+fileType
    if img_too_large: 
      print ("Image is too large to resize so preprocessed image is saved at {} for".format(newName),
             "future processing purposes")
    if fileType == ".tif":
      tifffile.imsave(newName, img)
    else:
      cv2.imwrite(newName,img)

  img = img.astype(np.float32) / float(np.max(img))

  if maskImg is not None:
    return img, maskImg
  else:
    return img

def processMask(degreesOffCenter,resizeScale, fileName = None, maskImg = None, imgShape = 1):
  '''
  function to reorient and resize the mask that was generated for the original
  image.
  '''
  maskName = fileName[:-4]+"_mask"+fileName[-4:]
  # mask = util.ReadImg(maskName)
  if img.shape < img_size_cutoff:
    reoriented = imutils.rotate_bound(maskImg,degreesOffCenter)
  else:
    reoriented = maskImg
  resized = cv2.resize(reoriented,None,fx=resizeScale,fy=resizeScale,interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(fileName[:-4]+"_processed_mask"+fileName[-4:],resized)

  return resized

def preprocessTissue():
  '''
  Function to preprocess the original tissue level image. This is not a general routine for 
  preprocessing any tissue image but one meant to standardize the preprocessing of the tissue
  image used in the manuscript.
  '''

  #root = "/net/share/pmke226/DataLocker/cardiac/Sachse/171127_tissue/"
  root = "./myoimages/"
  fileName = "tissue.tif"


  ### read in image
  tissueImg = cv2.imread(root+fileName)
  tissueImg = cv2.cvtColor(tissueImg,cv2.COLOR_BGR2GRAY)

  ### rescale to filter size
  imgTwoSarcSize = 22
  filterTwoSarcSize = 25
  scale = float(filterTwoSarcSize) / float(imgTwoSarcSize)
  resizedTissueImg = cv2.resize(tissueImg,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  ### smooth the large image
  ## This caused a weird ringing effect so I'm opting to do this after the CLAHE
  #smoothedTissueImg = cv2.blur(resizedTissueImg,(3,3))

  ### applying a much more global CLAHE routine to kill dye imbalance
  tissueDims = np.shape(resizedTissueImg)
  claheTileSize = int(1./8. * tissueDims[0])
  CLAHEDtissueImg = applyCLAHE(resizedTissueImg,claheTileSize)

  ### smooth the CLAHED image
  kernelSize = (3,3)
  smoothedTissueImg = cv2.blur(CLAHEDtissueImg,kernelSize)

  ### apply an intensity ceiling and floor to apply contrast stretching
  floorValue = 6
  ceilingValue = 10
  clippedTissueImg = smoothedTissueImg
  clippedTissueImg[clippedTissueImg < floorValue] = floorValue
  clippedTissueImg[clippedTissueImg > ceilingValue] = ceilingValue
  clippedTissueImg -= floorValue
  clippedTissueImg = clippedTissueImg.astype(np.float32)
  clippedTissueImg *= 255. / np.max(clippedTissueImg)
  clippedTissueImg = clippedTissueImg.astype(np.uint8)

  ### save image
  cv2.imwrite("./myoimages/preprocessedTissue.png",clippedTissueImg)

def preprocessAll():
  '''
  function meant to preprocess all of the images needed for data reproduction
  '''
  root = './myoimages/'
  imgNames = ["HF_1.png", 
              "MI_D_73.png",
              "MI_D_76.png",
              "MI_D_78.png",
              "MI_M_45.png",
              "MI_P_16.png",
              "Sham_11.png"]
  for name in imgNames:
    filterTwoSarcomereSize = 25
    # perform preprocessing on image
    preprocess(root+name,filterTwoSarcomereSize)
  
  ### now we preprocess the tissue image
  preprocessTissue()

def preprocessDirectory(directoryName,filterTwoSarcomereSize=25):
  '''
  function for preprocessing an entire directory of images
  '''
  for name in os.listdir(directoryName):
    preprocess(directoryName+name,filterTwoSarcomereSize)


###############################################################################
###
### Execution of File
###
###############################################################################

#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
tag = "default_"
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):

    ### Main routine to run
    if (arg=="-preprocess"):
      fileName = str(sys.argv[i+1])
      try:
        filterTwoSarcomereSize = sys.argv[i+2]
      except:
        filterTwoSarcomereSize = 25
      preprocess(fileName,filterTwoSarcomereSize)
      quit()
    if (arg=="-preprocessTissue"):
      preprocessTissue()
      quit()
    if (arg=="-preprocessAll"):
      preprocessAll()
      quit()
    if (arg=="-preprocessDirectory"):
      directory = str(sys.argv[i+1])
      preprocessDirectory(directory)
      quit()
      
  raise RuntimeError("Arguments not understood")


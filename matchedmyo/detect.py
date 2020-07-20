from __future__ import print_function
"""
Wrapper for 'simplist' of MF calls 
Ultimately will be posted on athena
"""

import sys

import matplotlib.pylab as plt
import numpy as np
import yaml

import bankDetect as bD
import cv2
import display_util as du
import matchedFilter as mf
import optimizer
import painter
import preprocessing as pp
import util
import myocyteFigs as myoF


def DisplayHits(img,threshed,
                smooth=8 # px
                ):
        # smooth out image to make it easier to visualize hits 
        daround=np.ones([smooth,smooth])
        sadf=mf.matchedFilter(threshed,daround,parsevals=False,demean=False)

        # merge two fields 
        #du.StackGrayRedAlpha(img,sadf,alpha=0.5)
        du.StackGrayBlueAlpha(img,sadf,alpha=0.5)


class empty:pass    
def lightlyPreprocess(img,filterTwoSarcomereSize):
  ### Lightly preprocess the image
  imgDims = np.shape(img)

  # grab subsection for resizing image. Extents are just guesses so they could be improved
  cY, cX = int(round(float(imgDims[0]/2.))), int(round(float(imgDims[1]/2.)))
  xExtent = 50
  yExtent = 25
  top = cY-yExtent; bottom = cY+yExtent; left = cX-xExtent; right = cX+xExtent
  indexes = np.asarray([top,bottom,left,right])
  subsection = np.asarray(img[top:bottom,left:right],dtype=np.float64)
  subsection /= np.max(subsection)
  img, scale, newIndexes = pp.resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes)
 
  # intelligently threshold image using gaussian thresholding
  img = pp.normalizeToStriations(img, newIndexes, filterTwoSarcomereSize)
  img = np.asarray(img,dtype=np.float64)
  img /= np.max(img)
  return img

def docalc(img,
           mf,
           maskName=None,
           lobemf=None,
           #corrThresh=0.,
           #s=1.,
           paramDict = optimizer.ParamDict(),
           debug=False,
           smooth = 8, # smoothing for final display
           iters = [-20,-10,0,10,20], # needs to be put into param dict
           fileName="corr.png"):



    ## Store info 
    inputs=empty()
    inputs.imgOrig = img
    inputs.mfOrig  = mf
    inputs.lobemf = lobemf

    #print "WARNING: TOO RESTRICTIVE ANGLES" 

    results = bD.DetectFilter(inputs,paramDict,iters=iters,display=debug)

    ### Read/Construct Mask
    if maskName != None:
      mask = util.ReadImg(maskName)
      binaryMask = mask.copy().astype(np.float32)
      binaryMask[binaryMask != np.max(binaryMask)] = 0.0
      binaryMask[binaryMask != 0] = 1.0

    pasteFilter = True
    if pasteFilter:

      MFy,MFx = util.measureFilterDimensions(mf)
      filterChannel = 0
      imgDim = np.shape(img)
      # creates a boolean placeholder for hits
      results.threshed = painter.doLabel(results,dx=MFx,dy=MFy,thresh=paramDict['snrThresh'])
      #coloredImageHolder[:,:,filterChannel] = filterChannelHolder
    
    if maskName != None:  
      ### Apply Mask
      results.threshed *= binaryMask

    #results.threshed[results.threshed != 0] = 255

    ### Construct colored image for display
    cImg = np.asarray((img.copy(),
                       img.copy(),
                       img.copy()),dtype=np.float32)
    cImg = np.rollaxis(cImg,0,start=3)
    cImg *= 255. #/ np.max(cImg)
    cImg = cImg.astype(np.uint8)

    ### Mark Results on Display Image
    # TTchannel is different since matplotlib has RGB color scheme
    TTchannel = 2
    cImg[:,:,TTchannel][results.threshed == 1] = 255

    print ("Writing file %s"%fileName)
    #plt.figure()
    #DisplayHits(img,results.threshed,smooth=smooth)
    plt.figure()
    plt.imshow(cImg)
    plt.gcf().savefig(fileName,dpi=300)


    return inputs,results 



#
# Simple performs a matched filtering detection with a single filter, image and threshold
#
def simple(imgName,mfName,filterTwoSarcomereSize,thresh,debug=False,smooth=4,outName="hits.png"): 
  img = util.ReadImg(imgName,renorm=True)
  mf  = util.ReadImg( mfName,renorm=True)

  paramDict = optimizer.ParamDict()
  paramDict['snrThresh'] = thresh
  paramDict['penaltyscale'] = 1.
  paramDict['useFilterInv'] = False
  paramDict['doCLAHE'] = False
  paramDict['demeanMF'] = False
  docalc(img,
         mf,
         paramDict = paramDict, 
         debug=debug,
         smooth=smooth, 
         fileName=outName)

#
# Calls 'simple' with yaml file 
# 
def simpleYaml(ymlName):
  with open(ymlName) as fp:
    data=yaml.load(fp)
  #print data    
  print ("Reading %s" % ymlName )
    
  if 'outName' in data:
      outName=data['outName']
  else:
      outName="hits.png"
  #print outName 
  
  simple(
    data['imgName'],
    data['mfName'],
    data['filterTwoSarcomereSize'],
    data['thresh'],
    debug= data['debug'],
    outName=outName)

###
### updated yaml call
###
def updatedSimpleYaml(ymlName,outFileName,imgFileName,maskFileName):
  print ("Adapt to accept filter mode argument?")
  with open(ymlName) as fp:
    data = yaml.load(fp)
  print ("Reading %s" % ymlName)

  mfName = "./myoimages/newSimpleWTFilter.png"
  
  updatedSimple(imgFileName,
                mfName,
                maskFileName,
                data['filterTwoSarcomereSize'],
                data['thresh'],
                debug=data['debug'],
                outName=outFileName
                )

###
###  Updated YAML routine to lightly preprocess image
###
def updatedSimple(imgName,
                  mfName,
                  maskName,
                  filterTwoSarcomereSize,
                  thresh,
                  debug=False,
                  smooth=4,
                  outName="hits.png"):
  '''
  Updated routine for the athena web server that utilizes WT punishment filter.

  INPUTS:
    - imgName: Name of image that has myocyte of interest in the middle of the image
               Myocyte major axis must be roughly parallel to x axis. There should be some conserved 
               striations in the middle of the image. 
    - thresh: Threshold that is utilized in the detection
  
  OUTPUTS:
    - None, writes image
  '''
  ### Load necessary inputs
  img = util.ReadImg(imgName)
  imgDims = np.shape(img)
  #mf = util.LoadFilter("./myoimages/newSimpleWTFilter")
  mf = util.LoadFilter(mfName)
  mfPunishment = util.LoadFilter("./myoimages/newSimpleWTPunishmentFilter.png")

  ### Lightly preprocess the image
  img = lightlyPreprocess(img,filterTwoSarcomereSize)

  ### Construct parameter dictionary
  paramDict = optimizer.ParamDict(typeDict="TT")
  paramDict['mfPunishment'] = mfPunishment
  paramDict['covarianceMatrix'] = np.ones_like(img)

  docalc(img,
         mf,
         maskName=maskName,
         paramDict=paramDict,
         debug=debug,
         iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
         smooth=smooth,
         fileName = outName
         )

def fullAnalysis(yamlFile,
                 outFileName,
                 imgFileName,
                 maskFileName=None,
                 root="/opt/webserver/matchedmyo/"
                 ):
  '''
  Function to perform the full analysis of an image based on TT, LT, and TA filtering
  Input:
    yamlFile -> str, name of the yaml file containing parameters to use in the analysis
    outFileName -> str, name of the image file that will be written. This is generateed by
                        the webserver
    imgFileName -> str, name of the image file to be analyzed, also generated by webserver
    maskFileName -> str or None, if str this is the name of the mask and the mask is applied.
                    if None, no mask is applied
  '''

  with open(yamlFile) as fp:
    data = yaml.load(fp)

  ### Assign non-default values if they are specified
  try:
    ttRotations = data['ttRotations'] 
  except:
    ttRotations = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
  try:
    ltRotations = data['ltRotations']
  except:
    ltRotations = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
  try:
    returnAngles = data['returnAngles']
  except:
    returnAngles = False
  try:
    ttGamma = data['ttGamma']
  except:
    ttGamma = None
  try:
    ttThresh = data['ttThresh']
  except:
    ttThresh = None
  try:
    ltThresh = data['ltThresh']
  except:
    ltThresh = None
  try:
    ltStdThresh = data['ltStdThresh']
  except:
    ltStdThresh = None
  try:
    taThresh = data['taThresh']
  except:
    taThresh = None
  try:
    taStdThresh = data['taStdThresh']
  except:
    taStdThresh = None

  ### Determine if there is a mask
  if maskFileName != None:
    masked = True
    mask = util.ReadImg(maskFileName)
    mask[mask != 0] = 1
  else:
    masked = False

  ### Create inputs class and store greyscale image
  inputs = empty()
  inputs.imgOrig = util.ReadImg(imgFileName)
  ## indicated we don't want to use a GPU
  inputs.useGPU = False
  
  ### Lightly preprocess the image
  inputs.imgOrig = lightlyPreprocess(inputs.imgOrig,data['filterTwoSarcomereSize'])

  ### Perform TT Filtering
  ttFilterName = root+'myoimages/newSimpleWTFilter.png'
  ttPunishFilterName = root+'myoimages/newSimpleWTPunishmentFilter.png'
  ttResults = myoF.WT_Filtering(inputs,ttRotations,ttFilterName,ttPunishFilterName,
                                ttThresh, ttGamma, returnAngles)
  ttStackedHits = ttResults.stackedHits

  ### Perform LT Filtering
  ltFilterName = root+'myoimages/LongitudinalFilter.png'
  ltResults = myoF.LT_Filtering(inputs,ltRotations,ltFilterName,ltThresh,ltStdThresh,returnAngles)
  ltStackedHits = ltResults.stackedHits

  ### Perform TA Filtering
  taFilterName = root+'myoimages/LossFilter.png'
  taResults = myoF.Loss_Filtering(inputs,taFilterName,taThresh,taStdThresh,returnAngles)
  taStackedHits = taResults.stackedHits

  ### Make original image 3 channel for color
  cImg = np.stack((inputs.imgOrig,
                   inputs.imgOrig,
                   inputs.imgOrig),axis=2)
  cImg /= np.max(cImg)
  cImg *= 255 *0.8 # 0.8 is to kill the brightness
  cImg = cImg.astype(np.uint8)

  ### Mark superthreshold hits for filters and convert to 8 bit format
  taStackedHits[taStackedHits != 0] = 255
  taStackedHits = taStackedHits.astype(np.uint8)
  ltStackedHits[ltStackedHits != 0] = 255
  ltStackedHits = ltStackedHits.astype(np.uint8)
  ttStackedHits[ttStackedHits != 0] = 255
  ttStackedHits = ttStackedHits.astype(np.uint8)

  ### Mask out hits so they don't overlap. Hierarchy of hits is TA -> LT -> WT
  ttStackedHits[taStackedHits == 255] = 0
  ltStackedHits[taStackedHits == 255] = 0
  ttStackedHits[ltStackedHits == 255] = 0

  ### Mask the stacked hits if applicable
  if masked:
    taStackedHits[masked == 0] = 0
    ltStackedHits[masked == 0] = 0
    ttStackedHits[masked == 0] = 0

  ### Run each stacked hits container through the paste filters routine to accurately show hits
  dummy = np.zeros_like(cImg)
  dummy = myoF.markPastedFilters(taStackedHits,ltStackedHits,ttStackedHits,dummy,
                                 lossName=taFilterName,
                                 ltName=ltFilterName,
                                 wtName=ttFilterName)
  ## apply mask if applicable
  if masked:
    for channel in np.shape(dummy)[2]:
      dummy[:,:,channel][mask == 0] = 0
  ## mark on image where hits are
  cImg[dummy == 255] = 255

  ### Save figure
  print ("Saving "+outFileName)
  plt.figure()
  plt.imshow(myoF.switchBRChannels(cImg))
  plt.gcf().savefig(outFileName,dpi=300)

def do2DGPUFiltering():
  '''
  Prototyping right now
  '''
  raise RuntimeError("GPU usage is deprecated.")

  import twoDtense as tdt
  import tissue

  inputs = empty()
  paramDict = optimizer.ParamDict(typeDict="TT")

  # grab image, store colored image, work with gray image
  tisParams = tissue.params
  case = empty()
  case.loc_um = [2577,279]
  case.extent_um = [250,250]
  case.cImg = util.ReadImg(tisParams.imgName,cvtColor=False)
  case.gray = cv2.cvtColor(case.cImg.copy(),cv2.COLOR_BGR2GRAY)
  tisParams.dim = np.shape(case.gray)
  tisParams.px_per_um = tisParams.dim/tisParams.fov
  case.gray = cv2.cvtColor(case.cImg.copy(),cv2.COLOR_BGR2GRAY)
  case.subregion = tissue.get_fiji(case.gray,case.loc_um,case.extent_um)
  img = case.subregion

  # preprocess image for algorithm
  import preprocessing as pp
  img,_ = pp.reorient(img)
  img,_,_,idxs = pp.resizeToFilterSize(img,25)
  img = pp.applyCLAHE(img,25)
  img2 = pp.normalizeToStriations(img,idxs,25)
  img2 /= np.max(img2)
  inputs.imgOrig = img2

  # 2D WT filters
  mf = util.LoadFilter('./myoimages/newSimpleWTFilter.png')
  mfPunishment = util.LoadFilter('./myoimages/newSimpleWTPunishmentFilter.png')
  inputs.mfOrig = mf
  # fix this dumb piece of code and store it inputs
  paramDict['mfPunishment'] = mfPunishment
  paramDict['covarianceMatrix'] = np.ones_like(img)

  # angles at which we'll do filtering
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]

  # call filtering code
  results,tElapsed = tdt.doTFloop(inputs,paramDict,ziters=iters)
  print (results.stackedHits)


  # crop image to subsection size
  loc = tissue.conv_fiji(case.loc_um[0],case.loc_um[1])
  d = tissue.conv_fiji(case.extent_um[0],case.extent_um[1])
  grayDims = np.shape(img)
  cSubregion = np.zeros((grayDims[0],grayDims[1],3))
  for i in range(3):
      #cSubregion[:,:,i] = case.cImg[loc[0]:(loc[0]+d[0]),loc[1]:(loc[1]+d[1]),i]
      cSubregion[:,:,i] = img
  #cSubregion = [case.cImg[loc[0]:(loc[0]+d[0]),loc[1]:(loc[1]+d[1]),channel] for channel in range(3)]
  holder = cSubregion[:,:,2]
  holder[results.stackedHits] = np.max(case.cImg)
  #case.cImg[:,:,results.stackedHits] = np.max(case.cImg)



  import matplotlib.pyplot as plt
  plt.figure()
  plt.imshow(results.stackedHits)
  plt.show()
  quit()


  #plt.imshow(coloredImage,cmap='gray')
  plt.figure()
  plt.imshow(cSubregion)
  plt.colorbar()
  plt.show()
  quit()

#
# Validation 
#
def validation(): 
  imgName ="myoimages/Sham_11.png"
  mfName ="myoimages/WTFilter.png"
  thresh = -0.08 
  thresh = 0.01
  print ("WHY IS THRESH NEGATIVE?"              )
  simple(imgName,mfName,thresh,smooth=20,
         debug=True)

  print ("WARNING: should add an assert here of some sort")

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
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # validation
    if(arg=="-validation"):             
      validation() 
      quit()

    # general test
    if(arg=="-simple"):             
      imgName =sys.argv[i+1] 
      mfName =sys.argv[i+2] 
      thresh = float(sys.argv[i+3])
      simple(imgName,mfName,thresh)
      quit()
  
    # general test with yaml
    if(arg=="-simpleYaml"):
      ymlName = sys.argv[i+1]
      outFileName = sys.argv[i+2]
      imgFileName = sys.argv[i+3]
      try:
        maskFileName = sys.argv[i+4]
      except:
        maskFileName = None
      updatedSimpleYaml(ymlName,outFileName,imgFileName,maskFileName)
      quit()

    if(arg=="-giveCorrelation"):
      imgName = sys.argv[i+1]
      mfName = sys.argv[i+2]
      thresh = float(sys.argv[i+3])
      debug = True
      simple(imgName,mfName,thresh,debug=debug)
      quit()

    if(arg=="-do2DTF"):
      do2DGPUFiltering()
      quit()





  raise RuntimeError("Arguments not understood")

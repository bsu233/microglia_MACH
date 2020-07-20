from __future__ import print_function
"""
For processing large tissue subsection from Frank
"""
# SPecific to a single case, so should be moved elsewhere
import sys
import matchedFilter as mF 
import numpy as np
import matplotlib.pylab as plt
import cv2
import util
import display_util as du
import imutils
import optimizer
import detection_protocols as dps
import detect
class empty:pass


## Based on image characteristics 
#params = empty()
#params.dim = np.shape(gray)
#params.fov = np.array([3916.62,4093.31]) # um (from image caption in imagej. NOTE: need to reverse spatial dimensions to correspond to the way cv2 loads image)
#params.px_per_um = params.dim/params.fov

class Params:pass
params = Params()
params.imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif"
#params.imgName = "tissue.tif"
params.fov = np.array([3916.62,4093.31]) # um (from image caption in imagej. NOTE: need to reverse spatial dimensions to correspond to the way cv2 loads image)


def Setup():
  img = cv2.imread(params.imgName)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  params.dim = np.shape(gray)
  params.px_per_um = params.dim/params.fov

  return gray

def SetupTest():
  case=empty()
#case.loc_um = [2366,1086] 
  case.loc_um = [2577,279] 
  case.extent_um = [150,150]
  SetupCase(case)
  return case
  
def SetupCase(case):
  case.orig = Setup()
  case.subregion = get_fiji(case.orig,case.loc_um,case.extent_um)

def SetupFilters(rot=20.):
  #mfr = CreateFilter(params,rot=rot)
  mfr = util.ReadImg('./myoimages/WTFilter.png',renorm=True)
  #lobemfr = CreateLobeFilter(params,rot=rot)
  lobemfr = None

  params.mfr=mfr
  params.lobemfr=lobemfr

def SetupParams():
  smoothScale=3 # number of pixels over which real TT should respond
  snrThresh = 12            
  lossScale = 10 # " over which loss region should be considered
  lossRegionCutoff = 40
  paramDict = optimizer.ParamDict(typeDict='TT')
  #paramDict = {
      #'doCLAHE':False,      
      #'useFilterInv':False,      
      #'sigma_n':1,
      #'filterMode':"lobemode",
  paramDict['smoothScale'] = smoothScale
      #'snrThresh':snrThresh,    
  paramDict['rawFloor'] = 1. # minimum value to image (usually don't chg)
  paramDict['eps'] = 4. # max intensity for TTs
  paramDict['lossScale'] = lossScale    
  paramDict['lossRegionCutoff'] = lossRegionCutoff
  paramDict['mfPunishment'] = util.ReadImg('./myoimages/WTPunishmentFilter.png',renorm=True)
  return paramDict



#print dim
#print px_per_um


# Functions for rescaling 'fiji' coordinates for image to those used by cv2
def conv_fiji(x_um,y_um): # in um
    y_px = int(x_um * params.px_per_um[1])
    x_px = int(y_um * params.px_per_um[0])
    return x_px,y_px

def get_fiji(gray, loc_um,d_um):
    loc= conv_fiji(loc_um[0],loc_um[1])
    d = conv_fiji(d_um[0],d_um[1])
    subregion = gray[loc[0]:(loc[0]+d[0]),loc[1]:(loc[1]+d[1])]
    subregionDim = np.shape(subregion) 
  
    print ("Extracting %dx%d region from %dx%d image"%(
      #d_um[0]*params.px_per_um[0],
      #d_um[1]*params.px_per_um[1],
      subregionDim[0]/params.px_per_um[0],
      subregionDim[1]/params.px_per_um[1],
      params.dim[0]/params.px_per_um[1],
      params.dim[1]/params.px_per_um[0]))


    return subregion

def dbgimg(case,results,
           rotIter=0, # which rotation result you want
           orig=True,
           thrsh=True,
           corr=True,
           corrLobe=True,
           snr=True,
           merged=True,
           mergedSmooth=True,
           ul=[0,0],
           lr=[100,100]
           ):
    l,r=ul
    ll,rr=lr

    rotResult = results.correlated[rotIter]
    
    if orig:
        plt.figure()
        plt.imshow(case.subregion[l:r,ll:rr],cmap="gray")
        plt.title("orig") 
        plt.colorbar()

    if thrsh:    
        plt.figure()
        plt.imshow(rotResult.img[l:r,ll:rr],cmap="gray")
        plt.title("thrsh") 
        plt.colorbar()

    if corr:
        plt.figure()
        plt.imshow(rotResult.corr[l:r,ll:rr],cmap="gray")
        plt.title("corr") 
        plt.colorbar()

    if corrLobe:    
        plt.figure()
        plt.imshow(rotResult.corrlobe[l:r,ll:rr],cmap="gray")
        plt.title("corrlobe") 
        plt.colorbar()

    if snr:
        plt.figure()
        plt.imshow(rotResult.snr[l:r,ll:rr],cmap="gray")
        plt.title("snr")   
        plt.colorbar()

    if merged:
        plt.figure()
        #stackem(results.corr[550:750,550:750],results.corrlobe[550:750,550:750])
        du.StackGrayRedAlpha(case.subregion[l:r,ll:rr],results.threshed[l:r,ll:rr])
        
        plt.title("threshed/marked") 

    if mergedSmooth:
        plt.figure()
        #sadf = sadf>50
        DisplayHits(case.subregion[l:r,ll:rr],results.threshed[l:r,ll:rr])
        plt.title("threshed/marked/smoothed") 
        plt.colorbar()     

def DisplayHits(img,threshed):
        # smooth out image to make it easier to visualize hits 
        daround=np.ones([40,40])
        sadf=mF.matchedFilter(threshed,daround,parsevals=False,demean=False)

        # merge two fields 
        du.StackGrayRedAlpha(img,sadf)
    
def CreateFilter(
    params,
#    rot = 30. # origin
    rot = 22. # hard
):  
    vert = int(3.5 * params.px_per_um[0])
    horz = int(2.1 * params.px_per_um[1])
    w = 2/2 # px (since tiling)
    mf = np.zeros([vert,horz])
    mf[:,0:w]=1.
    mf[:,(horz-w):horz]=1.

    # pad both sides (makes no diff)
    #pad = 2
    #mfp = np.zeros([vert,horz+2*pad])
    #mfp[:,pad:(horz+pad)] = mf
    #mf = mfp

    # multiple tiles
    mf = np.tile(mf, (1, 3))
    #imshow(mf,cmap='gray')


    #plt.figure()
    mfr = imutils.rotate_bound(mf,-rot)
    #imshow(mfr,cmap='gray')

    return mfr

def CreateLobeFilter(params,rot=22.):
  # test with embedded signal
  vert = int(3.5 * params.px_per_um[0])
  sigW = 4  # px
  lobeW = 4 # px
  lobemf = np.ones([vert,lobeW + sigW + lobeW])
  lobemf[:,lobeW:(lobeW+sigW)]=0.

#imshow(mf,cmap='gray')
  lobemfr = imutils.rotate_bound(lobemf,-rot)
  # imshow(lobemfr,cmap='gray')
  return lobemfr

def Test1(
  fileName = "test.png" 
  ):

  class empty:pass
  #cases = dict()
  
  ## Load in image of interest and identify region/px->um conversions
  #case=empty()
  #case.loc_um = [2366,1086] 
  #case.loc_um = [2577,279] 
  #case.extent_um = [150,150]
  #cases['hard'] = case
  case=SetupTest()

  results = Run(case,fileName=fileName)
  return results 
  
def Run(case,paramDict=None,fileName="out.png"):
  ## Load image 
  SetupCase(case)

  ## Set up matched filters to be used 
  SetupFilters()

  ## Define params  
  if paramDict is None:
    paramDict = SetupParams()

  ## Preprocessing
  # correct image so TT features are brightest
  # set noise floor 
  imgOrig = case.subregion
  img    = np.copy(imgOrig)              
  rawFloor = paramDict['rawFloor']
  img[imgOrig<rawFloor]=rawFloor        
  eps = paramDict['eps']
  img[ np.where(imgOrig> eps)] =eps # enhance so TT visible
  
  paramDict['covarianceMatrix'] = np.ones_like(imgOrig)

  
  ## do Matched filtering  
  inputs,results=detect.docalc(
                 case.subregion,
                 params.mfr,
                 lobemf=params.lobemfr,
                 debug=True,
                 iters=[0],
                 paramDict=paramDict)

  if fileName!=None: 
    plt.figure()
    DisplayHits(case.subregion,results.threshed)                 
    plt.gcf().savefig(fileName,dpi=300)

  
  # if orig. figure needed 
  #plt.figure()
  #plt.imshow(case.orig,cmap="gray")
  #plt.gcf().savefig("orig.png",dpi=300)
  return results 
  
  
  
  
#Test1()

def validate(): 
  results = Test1(fileName=None)

  # threshed contains matched filtering response 
  totInfo = np.sum( results.threshed ) 

  # assert 
  print ("WARNING: this test only verifies that the integrated total response is conserved - says nothing about accuracy" )
  truthVal = 2233257.
  assert( np.abs( totInfo - truthVal) < 1), "FAIL: %f != %f"%(totInfo,truthVal) 
  print ("PASS" )


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
  Tissue-based characterization of tissue 
 
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
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-validation"): 
      validate()
      quit()
    if(arg=="-test1"):
      Test1(fileName="tissuetest1.png")      
      quit()
  





  raise RuntimeError("Arguments not understood")





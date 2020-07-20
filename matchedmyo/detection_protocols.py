from __future__ import print_function
"""
Packages routines used to determine if correlation response
constitutes a detection
"""

import warnings
import numpy as np 
import matchedFilter as mF
import sys
import os
import util
import matplotlib.pylab as plt


###################################################################################################
###
### Class Definitions
###
###################################################################################################

class Results:
  '''This class contains all of the information that is necessary to keep for the results of 
  filtering
  '''

  def __init__(self,
               snr,
               corr,
               corrPunishment=None
               ):
    self.snr=snr
    self.corr=corr
    self.corrPunishment=corrPunishment

### Defining empty class for rapid prototyping
class empty:pass

###################################################################################################
###
### Filtering Functions
###
###################################################################################################

# This script determines detections by integrating the correlation response
# over a small area, then dividing that by the response of a 'lobe' filter 
# need to write this in paper, if it works 
def lobeDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
    # get data 
    img = inputs.imgOrig # raw (preprocessed image) 
    mf  = inputs.mf  # raw (preprocessed image) 
    lobemf  = inputs.lobemf  # raw (preprocessed image) 
    results = empty()

    ## get correlation plane w filter 
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=False)

    ## integrate correlation plane over XxX interval
    smoothScale = paramDict['smoothScale']
    smoother = np.ones([smoothScale,smoothScale])
    integrated =mF.matchedFilter(corr,smoother,parsevals=False,demean=False)

    ## get side lobe penalty
    if isinstance(lobemf,np.ndarray):
        #corrlobe = np.ones_like(corr)
        #out = mF.matchedFilter(corr,lobemf,parsevals=True,demean=False)
        #out -= np.min(out)
        #out /= np.max(out)
        #corrlobe += s*out
        print ("Needs work - something awry (negative numbes etc) ")
        corrlobe = mF.matchedFilter(corr,lobemf,parsevals=True,demean=False)
        corrlobe = np.ones_like(corr)
        
    else:    
        corrlobe = np.ones_like(corr)
        
    ## Determine SNR by comparing integrated area with corrlobe response 
    snr = integrated/corrlobe ##* corrThreshed
    print ("Overriding snr for now - NEED TO DEBUG" )
    snr = corr
    #snr = corrThreshed

    #plt.colorbar()
    #plt.gcf().savefig(name,dpi=300)

    ## make  loss mask, needs a threshold for defining maximum value a 
    ## region can take before its no longer a considered a loss region 
    lossScale = paramDict['lossScale']
    lossFilter = np.ones([lossScale,lossScale])
    losscorr = mF.matchedFilter(img,lossFilter,parsevals=False,demean=False)
    lossRegion = losscorr < paramDict['lossRegionCutoff']
    mask = 1-lossRegion
    results.lossFilter = mask


    
    ## 
    ## Storing 
    ## 
    results.img = img
    results.corr = corr
    results.corrlobe = corrlobe
    results.snr = snr  

    return results

# TODO phase this out 
def CalcInvFilter(inputs,paramDict,corr):
      penaltyscale = paramDict['penaltyscale'] 
      sigma_n  = paramDict['sigma_n']
      angle  = paramDict['angle']
      tN = inputs.imgOrig
      filterRef = inputs.mfOrig
      yP = corr
    
      s=1.  
      fInv = np.max(filterRef)- s*filterRef
      rFi = util.PadRotate(fInv,angle)
      rFiN = util.renorm(np.array(rFi,dtype=float),scale=1.)
      yInv  = mF.matchedFilter(tN,rFiN,demean=False,parsevals=True)   
      
      # spot check results
      #hit = np.max(yP) 
      #hitLoc = np.argmax(yP) 
      #hitLoc =np.unravel_index(hitLoc,np.shape(yP))

      ## rescale by penalty 
      # part of the problem earlier was that the 'weak' responses of the 
      # inverse filter would amplify the response, since they were < 1.0. 
      yPN =  util.renorm(yP,scale=1.)
      yInvN =  util.renorm(yInv,scale=1.)

      yPN = np.exp(yPN)
      yInvS = sigma_n*penaltyscale*np.exp(yInvN)
      scaled = np.log(yPN/(yInvS))
    
      return scaled 



# This script determines detections by correlating a filter with the image
# and dividing this response by a covariance matrix and a weighted 'punishment
# filter'
# need to write this in paper, if it works 
def punishmentFilter(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
    # get data 
    img = inputs.imgOrig # raw (preprocessed image) 
    mf  = inputs.mf  # raw (preprocessed image) 

    try:
      mfPunishment = paramDict['mfPunishmentRot']
    except:
      raise RuntimeError("No punishment filter was found in paramDict['mfPunishment']")
    try:
      cM = paramDict['covarianceMatrix']
    except:
      raise RuntimeError("Covariance matrix was not specified in paramDict")
    try:
      gamma = paramDict['gamma']
    except:
      raise RuntimeError("Punishment filter weighting term (gamma) not found\
                          within paramDict")

    ## get correlation plane w filter 
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=paramDict['demeanMF']) 
    corrPunishment = mF.matchedFilter(img,mfPunishment,parsevals=False,demean=False)

    ######
    snr = corr / (cM + gamma * corrPunishment)

    ### Store results
    results = Results(
      snr = snr,
      corr = corr,
      corrPunishment = corrPunishment
    )

    return results 

#
# Original detection procedure included with PNP paper
#
def simpleDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
  # get data 
  img = inputs.imgOrig # raw (preprocessed image) 
  mf  = inputs.mf  # raw (preprocessed image) 

  ## get correlation plane w filter 
  corr = mF.matchedFilter(img,mf,parsevals=False,demean=paramDict['demeanMF']) 
  
  if paramDict['useFilterInv']:
      snr = CalcInvFilter(inputs,paramDict, corr)

  else:
      snr = corr / paramDict['sigma_n']

  results = Results(
    snr = snr,
    corr = corr
  )

  return results

def regionalDeviation(inputs,paramDict):
  '''
  Detection protocol that will correlate the filter with the measured signal 
  and threshold based on simple threshold. From this list of hits, the 
  function will take the standard deviation of all pixels surrounding the hit
  on the original image. If this standard deviation is too high, signifying a 
  bright spot in the measured signal, then the hit will be discarded.
  '''

  ### Perform simple detection
  img = inputs.imgOrig
  mf = inputs.mf
  results = empty()

  if paramDict['useGPU'] == False:
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=paramDict['demeanMF'])
  else:
    raise RuntimeError("GPU Use is Deprecated")
    # corr = sMF.MF(img,mf,useGPU=True)


  ####### FINAL ITERATION OF CONVOLUTION BASED STD DEV
  ### Calculation taken from http://matlabtricks.com/post-20/calculate-standard-deviation-case-of-sliding-window

  ### find where mf > 0 to find elements in each window and construct kernel
  if isinstance(mf, list):
    kernel = [np.logical_not(np.equal(this_mf,0)).astype(float) for this_mf in mf]
  else:
    mfStdIdxs = np.nonzero(mf)
    kernel = np.zeros_like(mf)
    kernel[mfStdIdxs] = 1.

  ### construct array that contains information on elements in each window
  n = mF.matchedFilter(np.ones_like(img), kernel,parsevals=False,demean=paramDict['demeanMF'])

  ### Calculate Std Dev
  s = mF.matchedFilter(img,kernel,parsevals=False,demean=paramDict['demeanMF'])
  q = np.square(img)
  q = mF.matchedFilter(q, kernel,parsevals=False,demean=paramDict['demeanMF'])
  with warnings.catch_warnings() as w: # turn off errors due to NaNs cropping up (not an issue)
    warnings.simplefilter('ignore')
    stdDev = np.sqrt( np.divide((q-np.divide(np.square(s),n)),n-1)) 
  ## if s^2/n > q, we get NaN, so we can just convert that to zero here
  stdDev = np.nan_to_num(stdDev)

  ### Find common hits
  stdDevHits = stdDev < paramDict['stdDevThresh']
  if paramDict['inverseSNR'] == False:
    simpleHits = corr > paramDict['snrThresh']
  else:
    simpleHits = corr < paramDict['snrThresh']
  commonHits = np.multiply(stdDevHits, simpleHits)

  ### store in a resulting image with arbitrary snr
  if paramDict['inverseSNR'] == False:
    snr = np.zeros_like(img)
    snr[commonHits] = 5 * paramDict['snrThresh']
  else:
    snr = np.ones_like(img) * 5. * paramDict['snrThresh']
    snr[commonHits] = 0.
  
  results = Results(
    snr = snr,
    corr = corr
  )

  return results

def filterRatio(inputs,paramDict):
  # get data 
  img = inputs.imgOrig # raw (preprocessed image) 
  mf  = inputs.mf 
  mfPunish = paramDict['mfPunishment']  

  ## get correlation plane w filter 
  results = empty()
  if paramDict['useGPU'] == False:
    results.corr = mF.matchedFilter(img,mf,parsevals=False,demean=paramDict['demeanMF'])
    results.corrPunishment = mF.matchedFilter(img,mfPunish,parsevals=False,demean=paramDict['demeanMF'])
    #results.corr = sMF.MF(img,mf,useGPU=False)
  elif paramDict['useGPU'] == True:
    raise RuntimeError("GPU Use is Deprecated")
    # results.corr = sMF.MF(img,mf,useGPU=True)
    # results.corrPunishment = sMF.MF(img,mfPunish,useGPU=True)

  results.snr = results.corr  / results.corrPunishment

  return results

###################################################################################################
###
### Function That Routes Classification to Correct Filtering Function
###
###################################################################################################

def FilterSingle(
  inputs, # object with specific filters, etc needed for matched filtering
  paramDict = dict(),# pass in parameters through here
  mode = None
  ):

  '''This function calls different modes of selecting best hits
  '''

  if mode is not None:
      print ("WARNING: replacing with paramDict")
    
  mode = paramDict['filterMode']  
  if mode=="lobemode":
    results = lobeDetect(inputs,paramDict)
  elif mode=="punishmentFilter": 
    # for the WT SNR. Uses WT filter and WT punishment filter
    results = punishmentFilter(inputs,paramDict)
  elif mode=="simple":
    results = simpleDetect(inputs,paramDict)
  elif mode=="regionalDeviation":
    results = regionalDeviation(inputs,paramDict)
  elif mode=="filterRatio":
    results = filterRatio(inputs,paramDict)
  else: 
    #raise RuntimeError("need to define mode") 
    print ("Patiently ignoring you until this is implemented" )
    results = Results(
      snr = None,
      corr = None
    )

  return results




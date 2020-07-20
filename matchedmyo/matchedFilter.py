"""
Performs basic matched filter test by FFT-correlating a measured signal with an input signal/filter 
"""
import numpy as np
# import modules we'll need 
import scipy.fftpack as fftp
from scipy import ndimage
import cv2
from six import iteritems

##
## Basic steps of matched filtering process
## 
def matchedFilter(
  dimg, # imgs wherein signal (filter) is hypothesized to exist
  daFilter,# signal 
  parsevals=False,
  demean=True
  ):

  ### Check to see if the image is 2D or 3D. cv2 algorithm is a bit quicker for 2D but it is not compatible with 3D images
  if len(np.shape(dimg)) == 3:
    if isinstance(daFilter, list):
      # this indicates that we've decomposed the filter for sequential filtering
      h = sequential_filtering(dimg, daFilter)
    
    else:
      # Use scipy ndimage routine to convolve the 3D image and Filter
      h = ndimage.convolve(dimg,daFilter)#,mode='mirror')
  else:
    useCV2 = True
    if useCV2:
      if isinstance(daFilter, list):
        # this indicates that we've decomposed the filter for sequential filtering
        h = sequential_filtering(dimg, daFilter)

      else:
        h = cv2.filter2D(dimg,-1,daFilter)

    else:
      # placeholder for 'noise' component (will refine later)
      fsC = np.ones(np.shape(dimg))

      ## prepare img
      # demean/shift img
      if demean:
        sdimg = fftp.fftshift(dimg - np.mean(dimg))
      else:
        sdimg = fftp.fftshift(dimg)
      # take FFT 
      fsdimg = fftp.fft2(sdimg)

        ## zero-pad filter
      si = np.shape(dimg)
      sf = np.shape(daFilter)
      # add zeros
      zeropad = np.zeros(si)
      zeropad[:sf[0],:sf[1]]=daFilter
      # shift original ligand by its Nyquist
      szeropad = np.roll(\
        np.roll(zeropad,-sf[0]/2+si[0]/2,axis=0),-sf[1]/2+si[1]/2,axis=1)
      f= szeropad

      ## signal
      # shift,demean filter
      if demean:
        sfilter = fftp.fftshift(f- np.mean(f))
      else:
        sfilter = fftp.fftshift(f)
      # take FFT
      fsfilter= fftp.fft2(sfilter)

      ## matched filter
      fsh = fsdimg * fsfilter / fsC
      #fsh = np.real( fsh ) 
      sh = fftp.ifft2(fsh)
      h = fftp.ifftshift(sh)
      h = np.real(h)

      ## apply parsevals
      if parsevals:
        h *= 1/np.float(np.prod(np.shape(h)))
  return h 

def sequential_filtering(dimg,daFilter_list):
  '''Function to perform sequential filtering of 'dimg' given a list of filters.
  This is advantageous since there is a massive speedup for large 2D or 3D filters that can be 
  linearly decomposed.'''

  # loop through the filters and sequentially apply them
  # for (axis, filt) in iteritems(daFilter_dict):
  #   # check if it's a one-dimensional filter
  #   if isinstance(axis, int):
  #     dimg = ndimage.convolve1d(dimg, filt, axis=axis)
    
  #   elif isinstance(axis, str):
  #     dimg = ndimage.convolve(dimg, filt)
  for filt in daFilter_list:
    dimg = ndimage.convolve(dimg, filt)
    
  return dimg
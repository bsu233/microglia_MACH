"""
Performs basic matched filter test by FFT-correlating a measured signal with an input signal/filter 
"""
import numpy as np
# import modules we'll need 
import scipy.fftpack as fftp


##
## Basic steps of matched filtering process
## 
def matchedFilter(
  dimg, # imgs wherein signal (filter) is hypothesized to exist
  daFilter,# signal 
  parsevals=False,
  demean=False
  ):
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

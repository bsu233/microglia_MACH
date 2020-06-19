from __future__ import print_function
import sys
import matplotlib.pylab as plt 
import numpy as np 
import cv2
import scipy.fftpack as fftp
from scipy import ndimage
import imutils
import tifffile as tif
import numpy as np
from math import *

sys.path.append('./matchedmyo/')
import util

###################################################################################################
### Pete Routines
###################################################################################################

def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

# Prepare matrix of vectorized of FFT'd images
def CalcX(
  imgs,
  debug=False
  ):
  nImg,d1,d2 = np.shape(imgs)
  dd = d1*d2  
  #print nImg, d2
  X = np.zeros([nImg,dd],np.dtype(np.complex128))
    
  for i,img in enumerate(imgs):
    xi = np.array(img,np.dtype(np.complex128))     
    # FFT (don't think I need to shift here)  
    Xi = fftp.fft2( xi )    
    if debug:
      Xi = xi    
    #myplot(np.real(Xi))
    # flatten
    Xif = np.ndarray.flatten(Xi)
    X[i,:]=Xif
  return X  

def TestFilter(
  H, # MACE filter
  I  # test img
):
    #R = fftp.ifftshift(fftp.ifft2(I*conj(H)));
    icH = I * np.conj(H)
    R = fftp.ifftshift ( fftp.ifft2(icH) ) 
    #R = fftp.ifft2(icH) 

    daMax = np.max(np.real(R))
    print ("Response {}".format( daMax ))
    #myplot(R)
    return R,daMax

# renormalizes images to exist from 0-255
# rescale/renomalize image 
def renorm(img,scale=255):
    img = img-np.min(img)
    img/= np.max(img)
    img*=scale 
    return img

def GetAnnulus(region,sidx,innerMargin,outerMargin=None):
  if outerMargin==None: 
      # other function wasn't really an annulus 
      raise RuntimeError("Antiquated. See GetRegion")

  if innerMargin%2==0 or outerMargin%2==0:
      print ("WARNING: should use odd values for margin!" )
  #print "region shape", np.shape(region)

  # grab entire region
  outerRegion,dummy,dummy = GetRegion(region,sidx,outerMargin)
  #print "region shape", np.shape(outerRegion)

  # block out interior to create annulus 
  annulus = np.copy(outerRegion) 
  s = np.shape(annulus)
  aM = outerMargin - innerMargin
  xMin,xMax = 0+aM, s[0]-aM
  yMin,yMax = 0+aM, s[1]-aM
  interior = np.copy(annulus[xMin:xMax,yMin:yMax])
  annulus[xMin:xMax,yMin:yMax]=0. 

  return annulus,interior

def GetRegion(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area

def MaskRegion(region,sidx,margin,value=0):
      region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]=value  




###################################################################################################
### Dylan Functions
###################################################################################################

def do_edge_detection_on_image_list(img_list, sigma=2, verbose=False):
    '''This function takes a list of images, performs edge detection on the images using a gradient 
    magnitude using Gaussian derivatives and returns the gradient images in a new list
    
    Inputs:
        img_list -> list of images to perform edge detection on
        sigma -> standard deviation of the gaussian used to perform edge detections
        verbose -> Boolean flag to display the edge detected images
        
    Outputs:
        edges -> list of images that edge detection was ran on
    '''
    
    ### Do edge detection on all images
    # NOTE: Sigma = 2 might be too washed out but Sigma = 1 might be too noisy. Something worth playing around with
    
    edges = [ndimage.gaussian_gradient_magnitude(img, sigma=sigma) for img in img_list]

    if verbose:
        for i,img in enumerate(edges):
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(img_list[i],cmap='gray')
            ax1.set_title("Original Image")
            ax2.imshow(img,cmap='gray')
            ax2.set_title("Edge Detection of Image")
            plt.show()
    return edges

def generateAnnulusFilters(inner_radius, outer_radius,verbose=False):
    """This function serves as a filter generation routine for an annulus filter. This will be convenient when
    trying to differentiate hits from non-hits using peak-detection.
    
    This saves the annulus filters in the filter_images directory"""
    
    inner_annulus = np.ones((2*(inner_radius+1), 2*(inner_radius+1)),dtype=np.bool)
    outer_annulus = np.ones((2*(outer_radius+1), 2*(outer_radius+1)),dtype=np.bool)

    # get distance transforms
    inner_annulus[inner_annulus.shape[0]//2,inner_annulus.shape[1]//2] = 0
    outer_annulus[outer_annulus.shape[0]//2,outer_annulus.shape[1]//2] = 0
    inner_annulus = ndimage.morphology.distance_transform_edt(inner_annulus)
    outer_annulus = ndimage.morphology.distance_transform_edt(outer_annulus)

    # threshold based on distances from center
    inner_annulus = np.less(inner_annulus,inner_radius).astype(np.float32)
    outer_annulus = np.logical_and(
        np.greater_equal(outer_annulus, inner_radius),
        np.less(outer_annulus,outer_radius)
    ).astype(np.float32)

    # divide by the sum of each filters to normalize filtered intensity to 1
    inner_annulus /= np.sum(inner_annulus)
    outer_annulus /= np.sum(outer_annulus)

    name1 = "./filter_images/inner_annulus.tif"
    name2 = "./filter_images/outer_annulus.tif"
    
    tif.imsave(name1, inner_annulus)
    tif.imsave(name2, outer_annulus)
    
    if verbose:
      print ("Succesfully wrote {} and {}".format(name1,name2))

def giveAnnulusSNR(image, inner_annulus_radius, outer_annulus_radius, thresh, verbose=False):
    """Gives the signal-to-noise-ratio (SNR) of "image" defined by an annulus with dimensions given by inner_annulus_radius and 
    outer_annulus_radius. Thresholds the SNR given by the annulus based on the 'thresh' parameter.
    
    Returns:
      SNR -> numpy array defined by annulus SNR
      threshed -> boolean numpy array of annulus SNR determined by threshold"""
    
    # generate annulus filters
    generateAnnulusFilters(inner_annulus_radius,outer_annulus_radius,verbose=verbose)

    # load in the annulus filters
    inner = util.LoadFilter("./filter_images/inner_annulus.tif")
    outer = util.LoadFilter("./filter_images/outer_annulus.tif")

    # perform filtering 
    in_filt = ndimage.filters.convolve(image,inner)
    out_filt = ndimage.filters.convolve(image,outer)

    # calculate annulus response
    SNR = in_filt / out_filt

    # display annulus response
    if verbose:
        plt.figure()
        plt.imshow(SNR)
        plt.title("SNR Given by Annulus")
        plt.colorbar()

    # threshold said response
    threshed = np.greater(SNR,thresh)

    if verbose:
        plt.figure()
        plt.imshow(threshed,cmap='gray')
        plt.title("Thresholded SNR")
        
    return SNR, threshed

###################################################################################################
### Start Functions Written By Hadi
###################################################################################################

def add_rotated_imgs(list_of_images):
    '''It gets a list of images, rotate them with 90, 180 an 270 degrees, add them with original 
    photo list and returns a new list'''
    
    ### Give information about the original list
    print ("Length of original list:",len(list_of_images))
    
    ### Make a new list for the new images
    new_90 = []
    new_180 = []
    new_270 = []
    
    ### Loop through the old list, store old image plus the newly rotated images
    for original_img in list_of_images:
        # rotate the image 90 degrees
        rot_90 = imutils.rotate(original_img,90)

        # store the 90 degree rotated image
        new_90.append(rot_90)

        # rotate the image 180 degrees
        rot_180 = imutils.rotate(original_img,180)

        # store the 180 degree rotated image
        new_180.append(rot_180)

        # rotate the image 270 degrees
        rot_270 = imutils.rotate(original_img,270)

        # store the 270 degree rotated image
        new_270.append(rot_270)
        
    ### Concatenate all lists together to make new list
    total_images = new_90 + new_180 + new_270 + list_of_images
    print ("Length of the total_images list:",len(total_images))
    
    ### Return new list
    return total_images

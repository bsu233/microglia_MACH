#!/usr/bin/env python2
from __future__ import print_function
from six import iteritems

'''
The script is used serve as the wrapper of runing MACH filtering
'''

import os
import subprocess
import time
import sys
import datetime
import copy
import csv
import cv2
import util
import numpy as np
import optimizer
import bankDetect as bD
import matplotlib.pyplot as plt
import painter
import matchedFilter as mF
import argparse
import yaml
import preprocessing as pp
if sys.version_info[0] < 3:
  import cPickle as pkl
else:
  import pickle as pkl


root = '/'.join(os.path.realpath(__file__).split('/')[:-1])

###################################################################################################
###
### Class Definitions
###
###################################################################################################

class Inputs:
  '''Class for the storage of inputs for running through the classification routines,
  giveMarkedMyocyte and give3DMarkedMyocyte. This class is accessed at all levels of 
  characterization so it's convenient to have a way to pass parameters around.
  '''
  def __init__(self,
               classificationType = 'MACH',
               imageName = None,
               maskName = False,
               colorImage = None,
               yamlFileName = None,
               mfOrig=None,
               #scopeResolutions=None,
               efficientRotationStorage=True,
               paramDicts = None,
               yamlDict = None,
               preprocess = True
               ):
    '''
    Inputs:
      imageName -> Name of oringinal image that matched filtering will be performed on.
      yamlfileName -> str. Name of the yaml file for the characterization routine.
      mfOrig -> Original matched filter
      scopeResolutions -> Resolutions of the confocal microscope in pixels per micron. To be read in via YAML
      efficientRotationStorage -> Flag to tell the routine to use the newer, more efficient
                                    way to store information across rotations.
      *FilterName -> Names of the filters used in the matched filtering routines.
      ttFiltering, ltFiltering, taFiltering -> Flags to turn on or off the detection of TT, LT
                                                 and TA morphology.
      yamlDict -> dict. Dictionary read in by the setupYamlInputs() method. yamlFileName must be 
                    specified first.
    '''

    ### Store global class-level parameters
    self.classificationType = classificationType
    self.imageName = imageName
    self.maskName = maskName
    self.yamlFileName = yamlFileName
    self.mfOrig = mfOrig
    self.scopeResolutions = scopeResolutions
    self.efficientRotationStorage = efficientRotationStorage
    self.paramDicts = paramDicts
    self.preprocess = preprocess

    if sys.version_info[0] < 3:
      self.writeMode = 'wb'
    else:
      self.writeMode = 'w'

    ## Update default dictionaries according to yaml file
    if yamlFileName:
      self.load_yaml()
      try:
        self.classificationType = self.yamlDict['classificationType']
      except:
        pass
    else:
      self.yamlDict = {}

    ## Setup default dictionaries for classification
    self.setupDefaultDict()
    self.setupDefaultParamDicts()

    ## Update based on YAML file
    self.updateDefaultDict()
    self.setupImages()
    self.updateInputs()

    if self.yamlDict:
      self.check_yaml_for_errors()
    
  def setupImages(self):
    '''This function sets up the gray scale image for classification and the color image for marking
    hits.'''
    ### Read in the original image and determine number of dimensions from this
    self.imgOrig = util.ReadImg(self.dic['imageName'], renorm=True)
    self.dic['dimensions'] = len(self.imgOrig.shape)

    ## Turn on printing of filtering progress automatically if the image is 3D
    if self.dic['dimensions'] > 2:
      self.dic['displayProgress'] = True

    ### Make a 'color' image with 3 channels in the final index to represent the color channels
    ###   We also want to dampen the brightness a bit for display purposes, so we multiply by an
    ###   alpha value
    alpha = 0.85
    colorImageMax = 255
    self.colorImage = np.stack((self.imgOrig, self.imgOrig, self.imgOrig),axis=-1).astype(np.float32)
    self.colorImage *= alpha * colorImageMax
    self.colorImage = self.colorImage.astype(np.uint8)

    if isinstance(self.dic['maskName'], str):
      self.maskImg = util.ReadImg(self.dic['maskName'],renorm=True)
      self.maskImg[self.maskImg < np.max(self.maskImg)] = 0
      self.maskImg = self.maskImg.astype(np.uint8)
    else:
      self.maskImg = None

  def setupDefaultDict(self):
    '''This method sets up a dictionary to hold default classification inputs. This will then be 
    updated by updateInputs() method.'''
    dic = dict()
    
    ### Globabl parameters
    dic['classificationType'] = self.classificationType
    if isinstance(self.imageName, str):
      dic['imageName'] = self.imageName
    else:
      dic['imageName'] = ''
    dic['maskName'] = self.maskName
    if self.scopeResolutions == None:
      dic['scopeResolutions'] = {
        'x': None,
        'y': None,
        'z': None
      }
    else:
      dic['scopeResolutions'] = {
        'x': self.scopeResolutions[0],
        'y': self.scopeResolutions[1],
        'z': self.scopeResolutions[2]
      }
    dic['efficientRotationStorage'] = True
    dic['iters'] = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
    dic['returnAngles'] = False
    dic['returnPastedFilter'] = True
    dic['displayProgress'] = False

    if self.classificationType == 'arbitrary':
      ## We don't have a preprocessing routine defined for the arbitrary filtering case
      dic['preprocess'] = False
    elif self.classificationType == 'myocyte':
      ## But we do have a preprocessing routine defined for the myocyte cases
      dic['preprocess'] = self.preprocess
    
    dic['filterTwoSarcomereSize'] = 25

    ### Output parameter dictionary
    dic['outputParams'] = {
      'fileRoot': None,
      'fileType': 'png',
      'dpi': 300,
      'saveHitsArray': False,
      'csvFile': './results/classification_results.csv'
    }

    if self.classificationType != 'myocyte':
      ## Write to a different csv file
      dic['outputParams']['csvFile'] = './results/classification_results_arbitrary.csv'
    
    ### Filtering flags to turn on or off
    if self.classificationType == 'myocyte':
      dic['filterTypes'] = {
        'TT':True,
        'LT':True,
        'TA':True
      }
    elif self.classificationType == 'arbitrary':
      ## Setting default only filtering 1 turned on since we have to manually define parameter dictionaries
      dic['filterTypes'] = {
        'filter1':True,
        'filter2':False,
        'filter3':False
      }

    ### Store in the class
    self.dic = dic

  def updateDefaultDict(self):
    '''This method updates the default inputs dictionary formed from setupDefulatDict() method
    with the inputs specified in the yaml file.'''

    ### Iterate through all keys specified in yaml and figure out if they have a default value.
    ###   If they do, then we assign the non-default value specified in the yaml file in the 
    ###   dictionary we already formed.
    if isinstance(self.yamlDict, dict):
      for (key, value) in iteritems(self.yamlDict):
        ## Check to see if the key is pointing to the parameter dictionary. If it is, skip this
        ##   since we have functions that update it already
        if key == 'paramDicts':
          continue

        ## Check to see if the key is pointing to the outputParams dictionary specified in the YAML file
        if key == "outputParams":
          ## iterate through the dictionary specified in the yaml file and store non-default values
          for (outputKey, outputValue) in iteritems(value):
            if outputKey in self.dic['outputParams'].keys():
              self.dic['outputParams'][outputKey] = outputValue
            else:
              msg = str(outputKey)+" is not a parameter you can specify in the outputParams dictionary. Check your spelling"
              raise RuntimeError(msg)
          continue
        
        ## Here we check if the key is present within the default dictionary. If it is, we can then
        ##   see if a non-default value is specified for it.
        try:
          ## checking to see if a default value is specified
          if self.dic[key] != None:
            ## if it is, we store the non-default value(s)
            self.dic[key] = value
        except:
          ## if the key is not already specified in the default dictionary, then we continue on
          pass
    
    ## Update the image name
    if isinstance(self.yamlDict['imageName'],str):
      self.imageName = self.yamlDict['imageName']
    
    ## Update the mask name
    try:
      self.maskName = self.yamlDict['maskName']
    except:
      pass
    
    ## Check to see if '.csv' is present in the output csv file
    if (isinstance(self.dic['outputParams']['csvFile'],str) 
        and self.dic['outputParams']['csvFile'][-4:] != '.csv'):
      self.dic['outputParams']['csvFile'] = self.dic['outputParams']['csvFile'] + '.csv'

    ## Convert the scope resolutions into a list
    if isinstance(self.dic['scopeResolutions'], dict):
      scopeRes = [
        self.dic['scopeResolutions']['x'],
        self.dic['scopeResolutions']['y']
      ]
      try:
        scopeRes.append(self.dic['scopeResolutions']['z'])
      except:
        pass
      self.dic['scopeResolutions'] = scopeRes

    ## Flatten out iters if it is still a dictionary. This is necessary for 3D classification where
    ##   there are three axes of rotation
    if isinstance(self.dic['iters'], dict):
      flattenedIters = []
      for i in self.dic['iters']['x']:
        for j in self.dic['iters']['y']:
          for k in self.dic['iters']['z']:
            flattenedIters.append( [i,j,k] )
      self.dic['iters'] = flattenedIters

    ## Turn on filtering types if parameter dictionaries are specified even if they weren't specified to be turned on explicitly
    for (filterKey, filterToggle) in iteritems(self.dic['filterTypes']):
      if "paramDicts" in self.yamlDict.keys():
        if filterKey in self.yamlDict['paramDicts'].keys():
          # check to see if parameters are specified and it's turned off for some reason
          if "filterTypes" in self.yamlDict.keys():
            if self.yamlDict["filterTypes"][filterKey]:
              self.dic['filterTypes'][filterKey] = True
          # otherwise just turn it on since parameters are specified anyway
          else: self.dic['filterTypes'][filterKey] = True
 
  def setupDefaultParamDicts(self):
    '''This function forms the default parameter dictionaries for each filtering type, TT, LT, and 
    TA.'''
    ### Form dictionary that contains default parameters
    storageDict = dict()

    if self.classificationType == 'myocyte':
      filterTypes = ['TT','LT','TA']

      ### Assign default parameters
      storageDict['TT'] = optimizer.ParamDict(typeDict=filterTypes[0])
      storageDict['LT'] = optimizer.ParamDict(typeDict=filterTypes[1])
      storageDict['TA'] = optimizer.ParamDict(typeDict=filterTypes[2])
    elif self.classificationType == 'arbitrary':
      filterTypes = ['', '', '']

      ### Assign default parameters
      storageDict['filter1'] = optimizer.ParamDict(typeDict=filterTypes[0])
      storageDict['filter2'] = optimizer.ParamDict(typeDict=filterTypes[1])
      storageDict['filter3'] = optimizer.ParamDict(typeDict=filterTypes[2])

    self.paramDicts = storageDict

  def updateParamDicts(self):
    '''This function updates the parameter dictionaries previously formed in the method, 
    setupDefaultParamDicts() with the specifications in the yaml file.'''
    ## Default parameter dictionary is set to 2D, so if image is 3D, we need to update these
    if self.dic['dimensions'] == 3:
      ## Loop through and update each parameter dictionary
      for filteringType in self.paramDicts.keys():
        self.paramDicts[filteringType] = optimizer.ParamDict(typeDict=filteringType+'_3D')

    ## Iterate through and assign non-default parameters to correct dictionaries
    try:
      ## If this works, there are parameter options specified
      yamlParamDictOptions = self.yamlDict['paramDicts']
      for (filterType, paramDict) in iteritems(yamlParamDictOptions):
        ## Go through and assign all specified non-default parameters in the yaml file to the 
        ##   storageDict
        for (parameterName, parameter) in iteritems(paramDict):
          self.paramDicts[filterType][parameterName] = parameter

    except:
      # if the above doesn't work, then there are no parameter options specified and we can exit 
      return
    
  def updateInputs(self):
    '''This function updates the inputs class that's formed in matchedmyo.py script 
    
    Also updates parameteres based on parameters that are specified in the yaml dictionary 
    that is stored within this class.'''
    ### Form the correct default parameter dictionaries from this dimensionality measurement
    # self.updateDefaultDict()
    self.updateParamDicts()

    ### Check to see if we need to preprocess the image at all
    if self.dic['preprocess']:
      ## Catch 3D images since we don't have preprocessing routine for them yet
      if self.dic['dimensions'] > 2:
        raise RuntimeError("Preprocessing is not implemented for 3D images.")
      
      if self.maskImg is not None:
        self.imgOrig, self.maskImg = pp.preprocess(self.dic['imageName'], self.dic['filterTwoSarcomereSize'], self.maskImg, inputs=self)
      # pp.preprocess(self.dic['imageName'], self.dic['filterTwoSarcomereSize'], self.maskImg, inputs=self)
      else:  
        self.imgOrig = pp.preprocess(self.dic['imageName'], self.dic['filterTwoSarcomereSize'], self.maskImg, inputs=self)
      #  pp.preprocess(self.dic['imageName'], self.dic['filterTwoSarcomereSize'], self.maskImg, inputs=inputs)
      ## remake the color image
      eightBitImage = self.imgOrig.astype(np.float32).copy()
      eightBitImage = eightBitImage / np.max(eightBitImage) * 255. * 0.8 # 0.8 is to kill the brightness
      eightBitImage = eightBitImage.astype(np.uint8)
      self.colorImage = np.dstack((eightBitImage,eightBitImage,eightBitImage))

    ### Get a measure of the size of the image
    if isinstance(self.maskImg, np.ndarray):
      ## sum all elements of mask image where 1's indicate inside of mask
      self.dic['cell_size'] = float(np.sum(self.maskImg))
    else: self.dic['cell_size'] = float(np.prod(np.shape(self.imgOrig)))

    ### Catch returnAngles flag for 3D images
    # if self.dic['dimensions'] > 2 and self.dic['returnAngles']:
    #   raise RuntimeError("'returnAngles' is not yet implemented for 3D images. 'returnAngles' "
    #                      +"should be False in the input YAML file.")

  def load_yaml(self):
    '''Function to read and store the yaml dictionary'''
    self.yamlDict = util.load_yaml(self.yamlFileName)

  def check_yaml_for_errors(self):
    '''This function checks that the user-specified parameters read in through load_yaml() are valid'''
    ### Check if the classification types are valid
    if not any([self.classificationType == name for name in ['myocyte','arbitrary']]):
      raise RuntimeError('The classificationType specified is not valid. Double check this.')

    ### Check that the scope resolutions are specified correctly
    for value in self.dic['scopeResolutions']:
      if not isinstance(value, (float, int, type(None))):
        raise RuntimeError("Scope resolutions are not specified correctly. Ensure that the "
                           +"resolutions are integers, floats, or are left blank.")

    ### Check efficientRotationStorage
    if not isinstance(self.dic['efficientRotationStorage'], bool):
      raise RuntimeError("The efficientRotationStorage parameter is not a boolean type "
                         +"(True or False). Ensure that this is correct in the YAML file.")

    ### Check the rotations
    if not isinstance(self.dic['iters'], list):
      raise RuntimeError('Double check that iters is specified correctly in the YAML file')
    for value in self.dic['iters']:
      ## Check if the entries are lists (3D) or floats/ints (2D)
      if not isinstance(value, (float, int, list)):
          raise RuntimeError('Double check that the values specified for the rotations (iters) are '
                             +'integers or floats.')

    if not isinstance(self.dic['returnAngles'], bool):
      raise RuntimeError('Double check that returnAngles is either True or False in the YAML file.')

    if not isinstance(self.dic['returnPastedFilter'], bool):
      raise RuntimeError('Double check that returnPastedFilter is either True or False in the YAML file.')

    if not isinstance(self.dic['filterTwoSarcomereSize'], int):
      raise RuntimeError('Double check that filterTwoSarcomereSize is an integer.')

    ### Check output parameters
    if not isinstance(self.dic['outputParams']['fileRoot'], (type(None), str)):
      raise RuntimeError('Ensure that the fileRoot parameter in outputParams is either a string '
                         +'or left blank.')
    if not self.dic['outputParams']['fileType'] in ['png','tif','pdf']:
      raise RuntimeError('Double check that fileType in outputParams is either "png," "tif," or "pdf."')
    if not isinstance(self.dic['outputParams']['dpi'], int):
      raise RuntimeError('Ensure that dpi in outputParams is an integer.')
    if not isinstance(self.dic['outputParams']['saveHitsArray'], bool):
      raise RuntimeError('Ensure that saveHitsArray in outputParams is either True or False')
    if not isinstance(self.dic['outputParams']['csvFile'], str):
      raise RuntimeError('Ensure that csvFile in outputParams is a string.')

    ### Check that filter types is either true or false for all entries
    for (key, value) in iteritems(self.dic['filterTypes']):
      if not isinstance(value, bool):
        raise RuntimeError('Check that {} in filterTypes is either True or False'.format(key))

    if self.dic['returnAngles']:
      if self.dic['classificationType'] == 'myocyte':
        if not self.dic['filterTypes']['TT']:
          raise RuntimeError('TT filtering must be turned on if returnAngles is specified as True')

    ### Check that filter modes are specified correctly
    for (filtType, pDict) in iteritems(self.paramDicts):
      try:
        filtMode = pDict['filterMode']
      except:
        filtMode = False
      if filtMode != False:
        if filtMode not in ['simple', 'punishmentFilter', 'regionalDeviation'] and self.dic['filterTypes'][filtType]:
          raise RuntimeError('Check that filterMode for {} in your paramDicts'.format(filtType) 
                             +' is either "simple,"'
                             +'"punishmentFilter," or "regionalDeviation."')

  def setupYamlInputs(self):
    '''This function sets up inputs if a yaml file name is specified'''
    ### Check that the YAML file exists
    if not os.path.isfile(self.yamlFileName):
      raise RuntimeError("Double check that the yaml file that was specified is correct. Currently, "
                         +"the YAML file that was specified does not exist.")

    self.load_yaml()
    
    ### Double check that the image exists
    if not os.path.isfile(self.yamlDict['imageName']):
      raise RuntimeError('The specified image does not exist. Double-check that imageName is correct.')
    
    self.updateInputs()
    self.check_yaml_for_errors()

  def autoPreprocess(self):
    '''Function to automate preprocessing for rapid prototyping of the MatchedMyo software. This
    is meant for rough estimations of final classification and is not meant to replace the manual
    preprocessing done for figure-quality images.'''
    ### Lightly preprocess the image
    imgShape = self.imgOrig.shape

    ### Check data type of image. If it's not uint8, then we should rescale
    if self.imgOrig.dtype != np.uint8:
      self.imgOrig = self.imgOrig.astype(np.float32) / np.max(self.imgOrig) * 255.
      self.imgOrig = self.imgOrig.astype(np.uint8)

    # grab subsection for resizing image. Extents are just guesses so they could be improved
    cY, cX = int(round(float(imgShape[0]/2.))), int(round(float(imgShape[1]/2.)))
    xExtent = 50
    yExtent = 25
    top = cY-yExtent; bottom = cY+yExtent; left = cX-xExtent; right = cX+xExtent
    indexes = np.asarray([top,bottom,left,right])
    subsection = np.asarray(self.imgOrig[top:bottom,left:right],dtype=np.float64)
    subsection /= np.max(subsection)
    self.imgOrig, scale, newIndexes = pp.resizeGivenSubsection(
      self.imgOrig,
      subsection,
      self.dic['filterTwoSarcomereSize'],
      indexes
    )

    # If colorImg is supplied, then we resize that too
    if isinstance(self.colorImage,np.ndarray):
      self.colorImage = cv2.resize(self.colorImage, None, fx=scale, fy=scale,interpolation=cv2.INTER_CUBIC)

    if isinstance(self.maskImg,np.ndarray):
      self.maskImg = cv2.resize(self.maskImg, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # intelligently threshold image using gaussian thresholding
    self.imgOrig = pp.normalizeToStriations(
      self.imgOrig, 
      newIndexes, 
      self.dic['filterTwoSarcomereSize']
    )
    self.imgOrig = self.imgOrig.astype(np.float32) / float(np.max(self.imgOrig))

class ClassificationResults:
  '''This class holds all of the results that we will need to store.'''
  def __init__(self,
               inputs,
               markedImage=None,
               markedAngles=None,
               ttContent=None,
               ltContent=None,
               taContent=None,
               angleCounts=None
               ):
    '''
    Inputs:
      markedImage -> Numpy array. The image with TT, LT, and TA hits superimposed on the original image.
      markedAngles -> Numpy array. The image with TT striation angle color-coded on the original image.
    '''
    self.markedImage = markedImage
    self.markedAngles = markedAngles
    self.ttContent = ttContent
    self.ltContent = ltContent
    self.taContent = taContent
    self.angleCounts = angleCounts
  
    ### Check to make sure the CSV file specified as output for the classification results actually
    ###   exists and if it doesn't create it
    fileExists = os.path.isfile(inputs.dic['outputParams']['csvFile'])
    csv_output_dir = '/'.join(inputs.dic['outputParams']['csvFile'].split('/')[:-1])
    csv_output_dir_exists = os.path.isdir(csv_output_dir)
    ## Check to make sure the directory exists
    if not csv_output_dir_exists:
      msg = """
      The directory, "{}/", with which you specified the classification results to be 
      saved in the CSV file does not exist. Make sure to create this directory before 
      you run this program. Alternatively, change the outputParams:csvFile option in
      your input YAML file.""".format(csv_output_dir)
      raise RuntimeError (msg)
    else:
      ## Check to make sure the CSV file exists. Write a CSV file if it doesn't
      if not fileExists:
        with open(inputs.dic['outputParams']['csvFile'], inputs.writeMode) as csvFile:
          ## Create instance of writer object
          dummyWriter = csv.writer(csvFile)

          ## If the csv file did not already exists, we need to create headers for the output file
          header = [
            'Date of Classification',
            'Time of Classification',
            'Image Name',
            'TT Content',
            'LT Content',
            'TA Content',
            'Output Image Location and Root'
          ]
          dummyWriter.writerow(header)



  def writeToCSV(self, inputs):
    '''This function writes the results to a CSV file whose name is specified in the Inputs class 
    (Inputs.outputParams['csvFile'])'''

    if sys.version_info[0] < 3:
      appendMode = 'ab'
    else:
      appendMode = 'a'
    with open(inputs.dic['outputParams']['csvFile'], appendMode) as csvFile:
      ## Create instance of writer object
      dummyWriter = csv.writer(csvFile)
      
      ## Get Date and Time
      now = datetime.datetime.now()

      ## If the path specified for the output root is relative, get the absolute path
      if inputs.dic['outputParams']['fileRoot'][0] == '.': # indicates relative path
        real_path = root + inputs.dic['outputParams']['fileRoot'][1:]
      else:
        real_path = inputs.dic['outputParams']['fileRoot']

      ## Write the outputs of this classification to the csv file
      output = [
        now.strftime('%Y-%m-%d'),
        now.strftime('%H:%M:%S'),
        inputs.imageName,
        str(self.ttContent),
        str(self.ltContent),
        str(self.taContent),
        real_path
      ]

      ## Write outputs to csv file
      dummyWriter.writerow(output)

###################################################################################################
###
### Individual Filtering Routines
###
###################################################################################################

def TT_Filtering(inputs,
                 paramDict,
                 ):
  '''
  Takes inputs class that contains original image and performs WT filtering on the image
  '''
  print ("TT Filtering")
  start = time.time()

  ### Specify necessary inputs
  ## Read in filter
  ttFilter = util.LoadFilter(inputs.paramDicts['TT']['filterName'])
  inputs.mfOrig = ttFilter

  paramDict['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  paramDict['mfPunishment'] = util.LoadFilter(inputs.paramDicts['TT']['punishFilterName'])

  ### Perform filtering
  WTresults = bD.DetectFilter(
    inputs,
    paramDict,
    inputs.dic['iters'],
    returnAngles=inputs.dic['returnAngles']
  )  

  end = time.time()
  print ("Time for WT filtering to complete:",end-start,"seconds")

  return WTresults

def LT_Filtering(inputs,
                 paramDict,
                 ):
  '''
  Takes inputs class that contains original image and performs LT filtering on the image
  '''

  print ("LT filtering")
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.paramDicts['LT']['filterName'])

  ### Perform filtering
  LTresults = bD.DetectFilter(
    inputs,
    paramDict,
    inputs.dic['iters'],
    returnAngles=inputs.dic['returnAngles']
  )

  end = time.time()
  print ("Time for LT filtering to complete:",end-start,"seconds")

  return LTresults

def TA_Filtering(inputs,
                 paramDict,
                 iters=None,
                 ):
  '''
  Takes inputs class that contains original image and performs Loss filtering on the image
  '''
  print ("TA filtering")
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.paramDicts['TA']['filterName'])
  
  ## Check to see if iters (filter rotations are specified) if they aren't we'll just use 0 and 45
  ##   degrees since the loss filter is symmetric
  if iters != None:
    Lossiters = iters
  else:
    Lossiters = [0, 45] 
  
  ### Perform filtering
  Lossresults = bD.DetectFilter(
    inputs,
    paramDict,
    Lossiters,
    returnAngles=inputs.dic['returnAngles']
  )

  end = time.time()
  print ("Time for TA filtering to complete:",end-start,"seconds")

  return Lossresults

def analyzeTT_Angles(testImageName,
                     inputs,
                     ImgTwoSarcSize,
                     WTstackedHits,
                     ttFilterName = root+"/myoimages/newSimpleWTFilter.png"
                     ):
  '''This function analyzes the tubule striation angle for the transverse tubule filter.
  The routine does this by smoothing the original image with a small smoothing filter, constructing
  a TT filter with a larger field of view (longer tubules in the filter), and calculating the SNR 
  using this filter. Then, the routine uses the previously detected hits from the TT_Filtering()
  function to mask out the hits from the larger FOV filter. Teh reason this is necessary is due to 
  the fact that the original TT filter is very specific in terms of hits, but is extremely variable 
  in terms of what rotation the hit occurs at.
  
  Inputs:
    testImageName -> str. Name of the image that you are analyzing.
    inputs -> class. Inputs class already constructed in the giveMarkedMyocyte() function.
    iters -> list. List of iterations (rotations) at which we are analyzing filter response
    ImgTwoSarcSize -> int. Size of the filter/image two sarcomere size.
    WTstackedHits -> numpy array. Array where hits are marked as their SNR and non-hits are marked
                       as zero.
    ttFilterName -> str. Name of the transverse tubule filter used in this analysis.
  '''

  ### Read in original colored image
  #cImg = util.ReadImg(testImageName,cvtColor=False)

  ### perform smoothing on the original image
  dim = 5
  kernel = np.ones((dim,dim),dtype=np.float32)
  kernel /= np.sum(kernel)
  smoothed = mF.matchedFilter(inputs.imgOrig,kernel,demean=False)

  ### make longer WT filter so more robust to striation angle deviation
  ttFilter = util.LoadFilter(ttFilterName)
  longFilter = np.concatenate((ttFilter,ttFilter,ttFilter))
    
  rotInputs = copy.deepcopy(inputs)
  rotInputs.mfOrig = longFilter
  rotInputs.imgOrig = smoothed

  params = optimizer.ParamDict(typeDict='WT')
  params['snrThresh'] = 0 # to pull out max hit
  params['filterMode'] = 'simple' # we want no punishment since that causes high variation
    
  ### perform simple filtering
  smoothedWTresults = bD.DetectFilter(rotInputs,params,inputs.dic['iters'],returnAngles=True)
  smoothedHits = smoothedWTresults.stackedAngles

  ### Apply mask to the WTstackedHits so we don't have to apply it later and chop off part of the image
  WTstackedHits = util.ReadResizeApplyMask(
    WTstackedHits,
    testImageName,
    ImgTwoSarcSize,
    filterTwoSarcSize=ImgTwoSarcSize,
    maskName=inputs.dic['maskName'],
    maskImg = rotInputs.maskImg)

  ### pull out actual hits from smoothed results
  smoothedHits[WTstackedHits == 0] = 361

  coloredAnglesMasked = painter.colorAngles(inputs.colorImage,smoothedHits,inputs.dic['iters'])

  ### Check to see if we've used the new efficient way of storing information in the algorithm.
  ###   If we have, we already have the rotational information stored
  if rotInputs.efficientRotationStorage:
    angleCounts = smoothedHits.flatten()
  else:
    ## Otherwise, we have to go through and manually pick out rotations from their indexes in the 
    ##  iters list
    stackedAngles = smoothedHits
    dims = np.shape(stackedAngles)
    angleCounts = []
    for i in range(dims[0]):
      for j in range(dims[1]):
        rotArg = stackedAngles[i,j]
        if rotArg != -1:
          ### indicates this is a hit
          angleCounts.append(inputs.dic['iters'][rotArg])

  return angleCounts, coloredAnglesMasked

###################################################################################################
###
### Wrappers for Full Analysis of User-Supplied Images
###
###################################################################################################

def giveMarkedMyocyte(
      inputs,
      ImgTwoSarcSize=None,
      ):
  '''
  This function is the main workhorse for the detection of features in 2D myocytes.
    See give3DMarkedMyocyte() for better documentation.
    TODO: Better document this
  '''
  start = time.time()
  ### Create storage object for results
  myResults = ClassificationResults(inputs=inputs)

  ### Perform Filtering Routines
  ## Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    WTresults = TT_Filtering(
      inputs = inputs,
      paramDict = inputs.paramDicts['TT'],
    )
    WTstackedHits = WTresults.stackedHits
  else:
    #WTstackedHits = np.zeros_like(inputs.imgOrig)
    WTstackedHits = None

  ## Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs = inputs,
      paramDict = inputs.paramDicts['LT'],
    )
    LTstackedHits = LTresults.stackedHits
  else:
    # LTstackedHits = np.zeros_like(inputs.imgOrig)
    LTstackedHits = None

  ## Tubule Absence Filtering
  if inputs.dic['filterTypes']['TA']:
    Lossresults = TA_Filtering(
      inputs=inputs,
      paramDict = inputs.paramDicts['TA'],
    )
    LossstackedHits = Lossresults.stackedHits
  else:
    # LossstackedHits = np.zeros_like(inputs.imgOrig)
    LossstackedHits = None
 
  ## Marking superthreshold hits for loss filter
  if inputs.dic['filterTypes']['TA']:
    LossstackedHits[LossstackedHits != 0] = 255
    LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

    ## applying a loss mask to attenuate false positives from WT and Longitudinal filter
    if inputs.dic['filterTypes']['TT']:
      WTstackedHits[LossstackedHits == 255] = 0
    if inputs.dic['filterTypes']['LT']:
      LTstackedHits[LossstackedHits == 255] = 0

  ## marking superthreshold hits for longitudinal filter
  if inputs.dic['filterTypes']['LT']:
    LTstackedHits[LTstackedHits != 0] = 255
    LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

    ## masking WT response with LT mask so there is no overlap in the markings
    if inputs.dic['filterTypes']['TT']:
      WTstackedHits[LTstackedHits == 255] = 0

  ## marking superthreshold hits for WT filter
  if inputs.dic['filterTypes']['TT']:
    WTstackedHits[WTstackedHits != 0] = 255
    WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ## apply preprocessed masks
  if inputs.dic['filterTypes']['TT']:
    if isinstance(inputs.maskImg,np.ndarray):
      wtMasked = util.applyMask(
        WTstackedHits,
        inputs.maskImg
      )
    else:
      wtMasked = WTstackedHits
  else:
    wtMasked = None

  if inputs.dic['filterTypes']['LT']:
    if isinstance(inputs.maskImg,np.ndarray):
      ltMasked = util.applyMask(
        LTstackedHits,
        inputs.maskImg
      )
    else:
      ltMasked = LTstackedHits
  else:
    ltMasked = None

  if inputs.dic['filterTypes']['TA']:
    if isinstance(inputs.maskImg,np.ndarray):
      lossMasked = util.applyMask(
        LossstackedHits,
        inputs.maskImg
      )
    else:
      lossMasked = LossstackedHits
  else:
    lossMasked = None

  ## Save the hits as full-resolution arrays for future use 
  if inputs.dic['outputParams']['saveHitsArray']:
    outDict = inputs.dic['outputParams']
    if inputs.dic['filterTypes']['TA']:
      util.saveImg(
        img=lossMasked, 
        inputs=inputs, 
        switchChannels=False, 
        fileName=outDict['fileRoot']+'_TA_hits',
        just_save_array=True
      )
    
    if inputs.dic['filterTypes']['LT']:
      util.saveImg(
        img=ltMasked,
        inputs=inputs,
        switchChannels=False,
        fileName=outDict['fileRoot']+'_LT_hits',
        just_save_array=True
      )

    if inputs.dic['filterTypes']['TT']:
      util.saveImg(
        img=wtMasked,
        inputs=inputs,
        switchChannels=False,
        fileName=outDict['fileRoot']+'_TT_hits',
        just_save_array=True
      )

  if not inputs.dic['returnPastedFilter']:
    ### create holders for marking channels
    markedImage = inputs.colorImage.copy()
    WTcopy = markedImage[:,:,0]
    LTcopy = markedImage[:,:,1]
    Losscopy = markedImage[:,:,2]

    ### color corrresponding channels
    if inputs.dic['filterTypes']['TT']:
      WTcopy[wtMasked > 0] = 255
    if inputs.dic['filterTypes']['LT']:
      LTcopy[ltMasked > 0] = 255
    if inputs.dic['filterTypes']['TA']:
      Losscopy[lossMasked > 0] = 255

    ### mark mask outline on myocyte
    myResults.markedImage = util.markMaskOnMyocyte(
      markedImage,
      inputs.yamlDict['imageName']
    )

    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      ### mark mask outline on myocyte
      cI_written = myResults.markedImage

      ### write output image
      util.saveImg(
        img = cI_written,
        inputs = inputs
      )

  else:
    ## Mark filter-sized unit cells on blank image to represent hits
    myResults.markedImage = util.markPastedFilters(inputs, lossMasked, ltMasked, wtMasked)
    
    ### apply mask to marked image to avoid content > 1.0
    if isinstance(inputs.maskImg, np.ndarray):
      myResults.markedImage = util.applyMask(
        myResults.markedImage,
        inputs.maskImg
      )

    ## Superimpose hits onto image
    colorImgDummy = inputs.colorImage.copy()
    for i in range(3):
      colorImgDummy[...,i][myResults.markedImage[...,i] == 255] = 255
    myResults.markedImage = colorImgDummy

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    ### TODO: Make this an optional parameter specified in the YAML file
    # estimatedContent = util.estimateTubuleContentFromColoredImage(myResults.markedImage)
  
    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      ### mark mask outline on myocyte
      cI_written = myResults.markedImage

      ### write outputs	  
      # plt.figure()
      # plt.imshow(util.switchBRChannels(cI_written))
      # outDict = inputs.dic['outputParams']
      # plt.gcf().savefig(outDict['fileRoot']+"_output."+outDict['fileType'],dpi=outDict['dpi'])
      util.saveImg(
        img = cI_written,
        inputs = inputs
      )

  if inputs.dic['returnAngles']:
    myResults.angleCounts, myResults.markedAngles = analyzeTT_Angles(
      testImageName=inputs.yamlDict['imageName'],
      inputs=inputs,
      ImgTwoSarcSize=ImgTwoSarcSize,
      WTstackedHits=WTstackedHits
    )

    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      # plt.figure()
      # plt.imshow(util.switchBRChannels(myResults.markedAngles))
      # outDict = inputs.dic['outputParams']
      # plt.gcf().savefig(outDict['fileRoot']+"_angles_output."+outDict['fileType'],dpi=outDict['dpi'])
      util.saveImg(
        img = myResults.markedAngles,
        inputs = inputs,
        fileName = inputs.dic['outputParams']['fileRoot']+'_angles_output.'+inputs.dic['outputParams']['fileType']
      )
    
  ### Get accurate measure of feature content based on cell size
  ###   TODO: In later versions of the code, I'd like for this to be informed from an automatic mask segmentation
  myResults.ttContent = np.count_nonzero(myResults.markedImage[...,0] == 255) / inputs.dic['cell_size'] 
  myResults.ltContent = np.count_nonzero(myResults.markedImage[...,1] == 255) / inputs.dic['cell_size']   
  myResults.taContent = np.count_nonzero(myResults.markedImage[...,2] == 255) / inputs.dic['cell_size'] 

  ### Write results of the classification
  if isinstance(inputs.dic['outputParams']['csvFile'],str):
    myResults.writeToCSV(inputs=inputs)

  end = time.time()
  tElapsed = end - start
  print ("Total Elapsed Time: {}s".format(tElapsed))
  return myResults

def give3DMarkedMyocyte(
      inputs,
      ImgTwoSarcSize=None,
      returnAngles=False,
      returnPastedFilter=True,
      ):
  '''
  This function is for the detection and marking of subcellular features in three dimensions. 

  Inputs:
    testImage -> str. Name of the image to be analyzed. NOTE: This image has previously been preprocessed by 
                   XXX routine.
    scopeResolutions -> list of values (ints or floats). List of resolutions of the confocal microscope for x, y, and z.
    # tag -> str. Base name of the written files 
    xiters -> list of ints. Rotations with which the filters will be rotated about the x axis (yz plane)
    yiters -> list of ints. Rotations with which the filters will be rotated about the y axis (xz plane)
    ziters -> list of ints. Rotations with which the filters will be rotated about the z axis (xy plane)
    returnAngles -> Bool. Whether or not to return the angles with which the hit experienced the greatest SNR
                      NOTE: Do I need to delete this? Not like I'll be able to get a colormap for this
    returnPastedFilter -> Bool. Whether or not to paste filter sized unit cells where detections occur.
                            This translates to much more intuitive hit markings.
  
  Outputs:
    TBD
  '''
  start = time.time()

  ### Insantiate storage object for results
  myResults = ClassificationResults(inputs=inputs)

  ### Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    TTresults = TT_Filtering(
      inputs,
      paramDict = inputs.paramDicts['TT'],
      # returnAngles = returnAngles
    )
    TTstackedHits = TTresults.stackedHits
  else:
    TTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs,
      paramDict = inputs.paramDicts['LT'],
      # returnAngles = returnAngles
    )
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Tubule Absence Filtering
  if inputs.dic['filterTypes']['TA']:
    ## form tubule absence flattened rotation matrix. Choosing to look at tubule absence at one rotation right now.
    taIters = [[0,0,0]]
    TAresults = TA_Filtering(
      inputs,
      iters=taIters,
      paramDict = inputs.paramDicts['TA'],
      # returnAngles = returnAngles
    )
    TAstackedHits = TAresults.stackedHits
  else:
    TAstackedHits = np.zeros_like(inputs.imgOrig)

  ### Mark Detections on the Image
  cImg = np.stack(
    (
      inputs.imgOrig,
      inputs.imgOrig,
      inputs.imgOrig
    ),
    axis=-1
  )
  ## Scale cImg and convert to 8 bit for color marking
  alpha = 0.75
  cImg = cImg.astype(np.float)
  cImg /= np.max(cImg)
  cImg *= 255 * alpha
  cImg = cImg.astype(np.uint8)
  inputs.colorImage = cImg
  if inputs.dic['returnPastedFilter']:
    ## Use routine to mark unit cell sized cuboids around detections
    myResults.markedImage = util.markPastedFilters(inputs,
                             TAstackedHits,
                             LTstackedHits,
                             TTstackedHits,
                             ttName = inputs.paramDicts['TT']['filterName'],
                             ltName = inputs.paramDicts['LT']['filterName'],
                             taName = inputs.paramDicts['TA']['filterName']
                             )

    ### 'Measure' cell volume just by getting measure of containing array
    cellVolume = np.float(np.product(inputs.imgOrig.shape))

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    # estimatedContent = util.estimateTubuleContentFromColoredImage(
    #   myResults.markedImage,
    #   totalCellSpace=cellVolume,
    #   taFilterName=inputs.paramDicts['TA']['filterName'],
    #   ltFilterName=inputs.paramDicts['LT']['filterName'],
    #   ttFilterName=inputs.paramDicts['TT']['filterName']
    # )

  else:
    ## Just mark exactly where detection is instead of pasting unit cells on detections
    myResults.markedImage = inputs.colorImage.copy()

    myResults.markedImage[:,:,:,2][TAstackedHits > 0] = 255
    myResults.markedImage[:,:,:,1][LTstackedHits > 0] = 255
    myResults.markedImage[:,:,:,0][TTstackedHits > 0] = 255

  ### Normalize the number of detections to the cell area/volume
  ###   TODO: In later versions of the code, I'd like for this to be informed from an automatic mask segmentation
  myResults.ttContent = np.count_nonzero(myResults.markedImage[...,0]) / inputs.dic['cell_size'] 
  myResults.ltContent = np.count_nonzero(myResults.markedImage[...,1]) / inputs.dic['cell_size']   
  myResults.taContent = np.count_nonzero(myResults.markedImage[...,2]) / inputs.dic['cell_size'] 

  ### Mark the hits on the color image
  # myResults.colorImage = inputs.colorImage.copy()

  if returnAngles:
    print ("WARNING: Striation angle analysis is not yet available in 3D")
  
  ### Save detection image
  if isinstance(inputs.dic['outputParams']['fileRoot'], str):
    # util.Save3DImg(myResults.markedImage,inputs.dic['outputParams']['fileRoot']+'.'+inputs.dic['outputParams']['fileType'],switchChannels=True)
    util.saveImg(
      img = myResults.markedImage,
      inputs = inputs
    )

  ### Write results of the classification
  myResults.writeToCSV(inputs=inputs)

  end = time.time()
  print ("Time for algorithm to run:",end-start,"seconds")
  
  return myResults

def arbitraryFiltering(inputs):
  '''This function is for matched-filtering of arbitrary combinations of user-supplied images
  and filters with non-default parameter dictionaries.
  
  Inputs:
    inputs -> See Inputs Class
    
  Outputs:
    myResults -> See ClassificationResults class
  '''
  start = time.time()
  myResults = ClassificationResults(
    inputs = inputs,
    markedImage = inputs.colorImage.copy()
  )

  ### Loop over filtering types and perform classification for that filter type if turned on by YAML file
  for (filterKey, filterToggle) in iteritems(inputs.dic['filterTypes']):
    if filterToggle:
      print ("Performing {} classification".format(filterKey))
      ## Load in filter
      inputs.mfOrig = util.LoadFilter(inputs.paramDicts[filterKey]['filterName'])

      if inputs.paramDicts[filterKey]['filterMode'] == 'punishmentFilter':
        # We have to load in the punishment filter too
        inputs.paramDicts[filterKey]['mfPunishment'] = util.LoadFilter(
          inputs.paramDicts[filterKey]['punishFilterName']
        )
      ## Load punishment filter and covariance matrix if applicable
      if inputs.paramDicts[filterKey]['filterMode'] == 'punishmentFilter':
        inputs.paramDicts[filterKey]['punishmentFilter'] = util.LoadFilter(
          inputs.paramDicts[filterKey]['punishFilterName']
        )
        inputs.paramDicts[filterKey]['covarianceMatrix'] = np.ones_like(inputs.imgOrig)

      ## Perform filtering
      filterResults = bD.DetectFilter(
        inputs = inputs,
        paramDict = inputs.paramDicts[filterKey],
        iters = inputs.dic['iters'],
        returnAngles = inputs.dic['returnAngles']
      )

      ## Get the channel index of the filter
      channelIndex = int(filterKey[-1]) - 1

      ## Save the hits as full-resolution arrays for future use 
      if inputs.dic['outputParams']['saveHitsArray']:
        util.saveImg(
          img = filterResults.stackedHits,
          inputs = inputs,
          switchChannels=False,
          fileName = inputs.dic['outputParams']['fileRoot']+'_'+filterKey+'_hits'
        )
        if inputs.dic['dimensions'] == 2:
          util.saveImg(
            img = filterResults.stackedAngles,
            inputs = inputs,
            switchChannels=False,
            fileName = inputs.dic['outputParams']['fileRoot']+'_'+filterKey+'_hits_angles'
          )
        else:
          ## Get correct writing mode
          if sys.version_info[0] < 3: writeMode = 'w'
          else: writeMode = 'wb'
          with open(inputs.dic['outputParams']['fileRoot']+'_'+filterKey+'_hit_angles.pkl', writeMode) as f:
            pkl.dump(filterResults.stackedAngles, f)

      ## Mark hits on the colored image
      if inputs.dic['returnPastedFilter']:
        ## Read in filter dimensions
        # filtDims = util.measureFilterDimensions(inputs.mfOrig)

        ## Trying new marking scheme based on filter dilation
        filterResults.stackedHits = painter.doLabel_dilation(
          filterResults,
          inputs.mfOrig,
          inputs
        )

      print("Marked image dims:", np.shape(myResults.markedImage))
      print("StackedHits image dims:", np.shape(filterResults.stackedHits))
      myResults.markedImage[..., channelIndex][filterResults.stackedHits != 0] = 255

      ## Measure amount of hits in image relative to image size
      hitRatio = float(np.count_nonzero(filterResults.stackedHits)) / float(np.prod(inputs.imgOrig.shape))
      print (filterKey+' detection to non-detection ratio: '+str(hitRatio)[:7])
      # storage{filterKey} = hitRatio

      ## Save angles of detection if indicated
      if inputs.dic['returnAngles']:
        ## This following is only implemented in 2D
        if inputs.dic['dimensions'] == 2:
          ## NOTE: not saving in myResults since we'd overwrite during each iteration of loop
          coloredAngles = painter.colorAngles(inputs.colorImage.copy(),filterResults.stackedAngles,inputs.dic['iters'])

          if isinstance(inputs.dic['outputParams']['fileRoot'],str):
            # plt.figure()
            # plt.imshow(util.switchBRChannels(coloredAngles))
            # outDict = inputs.dic['outputParams']
            # plt.gcf().savefig(outDict['fileRoot']+'_'+filterKey+'_angles_output.'+outDict['fileType'],dpi=outDict['dpi'])
            util.saveImg(
              img = coloredAngles,
              inputs = inputs,
              fileName = inputs.dic['outputParams']['fileRoot']+filterKey+'_angles_output.'+inputs.dic['outputParams']['fileType']
            )

        ## Count num hits at each rotation
        angleCounts = {}
        for it in inputs.dic['iters']:
          if isinstance(it,list):
            print ("The counting of angles hits for 3D images is not yet supported.")
            itCopy = [str(thisIt) for thisIt in it]
            key = '_'.join(itCopy)
          else:
            key = it
          angleCounts[key] = np.count_nonzero(filterResults.stackedAngles == it)
          print (filterKey+' hits at {} rotation: {}'.format(key, angleCounts[key]))

        

  ## Save image if indicated in inputs
  if isinstance(inputs.dic['outputParams']['fileRoot'], str):
    # plt.figure()
    # plt.imshow(util.switchBRChannels(myResults.markedImage))
    # outDict = inputs.dic['outputParams']
    # plt.gcf().savefig(outDict['fileRoot']+"_output."+outDict['fileType'],dpi=outDict['dpi'])
    util.saveImg(
        img = myResults.markedImage,
        inputs = inputs,
      )

  ### Write results of the classification
  myResults.writeToCSV(inputs=inputs)


  end = time.time()
  tElapsed = end - start
  print ("Total Elapsed Time: {}s".format(tElapsed))
  return myResults


###################################################################################################
###
###  Validation Routines
###
###################################################################################################

def fullValidation(args):
  '''This routine wraps all of the written validation routines.
  This should be run EVERYTIME before changes are committed and pushed to the repository.
  '''

  validate(args)
  validate3D(args)
  validate3D_arbitrary(args)

  print ("All validation tests have PASSED! MatchedMyo is installed properly on this machine.")
  print ("Happy classifying!")

def validate(args,
             display=False,
             capture_outputs = True
             ):
  '''This function serves as a validation routine for the 2D functionality of this repo.
  
  Inputs:
    display -> Bool. If True, display the marked image
  '''
  if capture_outputs:
    ### Capture all print statements
    sys.stdout = open('garbage.txt', 'w')

  ### Specify the yaml file NOTE: This will be done via command line for main classification routines
  yamlFile = './YAML_files/validate.yml'

  ### Setup inputs for classification run
  inputs = Inputs(
    yamlFileName = yamlFile
  )

  ### Run algorithm to pull out content and rotation info
  myResults = giveMarkedMyocyte(
    inputs=inputs
  )

  if display:
    plt.figure()
    plt.imshow(myResults.markedImage)
    plt.show()

  print ("\nThe following content values are for validation purposes only.\n")

  ### Calculate TT, LT, and TA content  
  ttContent, ltContent, taContent = util.assessContent(myResults.markedImage)

  if capture_outputs:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    subprocess.call(['rm', 'garbage.txt'])

  assert(abs(ttContent - 103050) < 1), "TT validation failed."
  assert(abs(ltContent -  68068) < 1), "LT validation failed."
  assert(abs(taContent - 156039) < 1), "TA validation failed."

  ### Calculate the number of hits at rotation equal to 5 degrees
  numHits = np.count_nonzero(np.asarray(myResults.angleCounts) == 5)
  # print "Number of Hits at Rotation = 5 Degrees:", numHits
  assert(abs(numHits - 1621) < 1), "Rotation validation failed"

  print ("\n2D Validation has PASSED!")

def validate3D(args, capture_outputs=True):
  '''This function serves as a validation routine for the 3D functionality of this repo.

  Inputs:
    None
  '''
  if capture_outputs:
    ### Capture all print statements
    sys.stdout = open('garbage.txt', 'w')

  ### Specify the yaml file. NOTE: This will be done via command line for main classification routines
  yamlFile = './YAML_files/validate3D.yml'

  ### Define parameters for the simulation of the cell
  ## Probability of finding a longitudinal tubule unit cell
  ltProb = 0.3
  ## Probability of finding a tubule absence unit cell
  taProb = 0.3
  ## Amplitude of the Guassian White Noise
  noiseAmplitude = 0.
  ## Define scope resolutions for generating the filters and the cell. This is in x, y, and z resolutions
  scopeResolutions = [10,10,5] #[vx / um]
  ## x, y, and z Dimensions of the simulated cell [microns]
  cellDimensions = [10, 10, 20]
  ## Define test file name
  testName = "./myoimages/3DValidationData.tif"
  ## Give names for your filters. NOTE: These are hardcoded in the filter generation routines in util.py
  # ttName = './myoimages/TT_3D.tif'
  # ttPunishName = './myoimages/TT_Punishment_3D.tif'
  # ltName = './myoimages/LT_3D.tif'
  # taName = './myoimages/TA_3D.tif'

  ### Simulate the small 3D cell
  util.generateSimulated3DCell(LT_probability=ltProb,
                               TA_probability=taProb,
                               noiseAmplitude=noiseAmplitude,
                               scopeResolutions=scopeResolutions,
                               cellDimensions=cellDimensions,
                               fileName=testName,
                               seed=1001,
                               )

  ### Setup input parameters for classification
  inputs = Inputs(
    yamlFileName = yamlFile
  )
  print (inputs.dic['outputParams']['fileRoot'], inputs.dic['outputParams']['fileType'])
  ### Analyze the 3D cell
  myResults = give3DMarkedMyocyte(
    inputs = inputs
  )

  print ("\nThe following content values are for validation purposes only.\n")

  ### Assess the amount of TT, LT, and TA content there is in the image 
  ttContent, ltContent, taContent = util.assessContent(myResults.markedImage)

  if capture_outputs:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    subprocess.call(['rm', 'garbage.txt'])

  ### Check to see that they are in close agreement with previous values
  ###   NOTE: We have to have a lot of wiggle room since we're generating a new cell for each validation
  assert(abs(ttContent - 305247) < 1), "TT validation failed."
  assert(abs(ltContent -  48624) < 1), "LT validation failed."
  assert(abs(taContent - 409161) < 1), "TA validation failed."
  print ("\n3D Validation has PASSED!")

def validate3D_arbitrary(args, capture_outputs=True):
  '''This function serves as a validation routine for the arbitrary classification functionality of
  this repo.

  Inputs:
    None
  '''
  if capture_outputs:
    ### Capture all print statements
    sys.stdout = open('garbage.txt', 'w')

  ### Specify the yaml file. NOTE: This will be done via command line for main classification routines
  yamlFile = './YAML_files/validate3D_arbitrary.yml'
  
  ### Define test file name
  testName = "./myoimages/3DValidationData.tif"

  if not os.path.isfile(testName):
    ### Define parameters for the simulation of the cell
    ## Probability of finding a longitudinal tubule unit cell
    filter2_Prob = 0.3
    ## Probability of finding a tubule absence unit cell
    filter3_Prob = 0.3
    ## Amplitude of the Guassian White Noise
    noiseAmplitude = 0.
    ## Define scope resolutions for generating the filters and the cell. This is in x, y, and z resolutions
    scopeResolutions = [10,10,5] #[vx / um]
    ## x, y, and z Dimensions of the simulated cell [microns]
    cellDimensions = [10, 10, 20]

    ### Simulate the small 3D cell
    util.generateSimulated3DCell(LT_probability=filter2_Prob,
                                 TA_probability=filter3_Prob,
                                 noiseAmplitude=noiseAmplitude,
                                 scopeResolutions=scopeResolutions,
                                 cellDimensions=cellDimensions,
                                 fileName=testName,
                                 seed=1001,
                                 )

  ### Setup input parameters for classification
  inputs = Inputs(
    yamlFileName = yamlFile
  )

  ### Analyze the 3D cell
  myResults = arbitraryFiltering(
    inputs = inputs
  )

  print ("\nThe following content values are for validation purposes only.\n")

  ### Assess the amount of TT, LT, and TA content there is in the image 
  filter1_Content, filter2_Content, filter3_Content = util.assessContent(myResults.markedImage)

  if capture_outputs:
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    subprocess.call(['rm', 'garbage.txt'])

  ### Check to see that they are in close agreement with previous values
  # These values are different from the 3D validation case using give3DMarkedMyocyte() since we 
  # don't shift the results down one spot
  assert(abs(filter1_Content - 242448) < 1), "Filter 1 validation failed."
  assert(abs(filter2_Content -  50112) < 1), "Filter 2 validation failed."
  assert(abs(filter3_Content - 412595) < 1), "Filter 3 validation failed."
  print ("\nArbitrary 3D Validation has PASSED!\n")

###################################################################################################
###
### Command Line Functionality
###
###################################################################################################

def run(args):
  '''This runs the main classification routines from command line
  '''
  ### Setup the inputs class
  inputs = Inputs(
    yamlFileName=args.yamlFile
  )

  ### Determine if classification is 2D or 3D and run the correct routine for it
  dim = len(np.shape(inputs.imgOrig))
  if inputs.dic['classificationType'] == 'myocyte':
    if dim == 2:
      giveMarkedMyocyte(inputs = inputs)
    elif dim == 3:
      give3DMarkedMyocyte(inputs = inputs)
    else:
      raise RuntimeError("The dimensions of the image specified in {} is not supported.".format(args.yamlFile))
  elif inputs.dic['classificationType'] == 'arbitrary':
    arbitraryFiltering(inputs = inputs)
  else:
    raise RuntimeError("Classification Type (specified as classificationType: <type> in YAML file)"
                       +"not understood. Check to see that spelling is correct.")

def main(args):
  '''The routine through which all command line functionality is routed.
  '''
  ### Get a list of all function names in the script
  functions = globals()

  functions[args.functionToCall](args)


### Begin argument parser for command line functionality IF function is called via command line
if __name__ == "__main__":
  description = '''This is the main script for the analysis of 2D and 3D confocal images of 
  cardiomyocytes and cardiac tissues.
  '''
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('functionToCall', 
                      type=str,
                      help='The function to call within this script')
  parser.add_argument('--yamlFile',
                      type=str,
                      help='The name of the .yml file containing parameters for classification')
  args = parser.parse_args()
  main(args)

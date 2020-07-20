#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cPickle as pkl

import bankDetect as bD
import detect
import display_util as du
import matchedFilter as mF
import optimizer
import painter
import preprocessing as pp
import tissue
import util
import matchedmyo as mm

##################################
#
# Revisions
#       July 24,2018 inception
#
##################################

class empty:pass

### Change default matplotlib settings to display figures
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 'large'
# print "Comment out for HESSE"
#plt.rcParams['axes.labelpad'] = 12.0 
plt.rcParams['figure.autolayout'] = True

#root = "myoimages/"
root = "/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"

###################################################################################################
###################################################################################################
###################################################################################################
###
### Figure Generation Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def WT_results(): 
  root = "/net/share/dfco222/data/TT/LouchData/processedMaskedNucleus/"
  testImage = root+"Sham_M_65_nucleus_processed.png"

  rawImg = util.ReadImg(testImage,cvtColor=False)

  iters = [-25,-20, -15, -10, -5, 0, 5, 10, 15, 20,25]

  ### Read in parameters from YAML file
  inputs = mm.Inputs(
    yamlFileName='./YAML_files/WT.yml'
  )
  
  coloredImg, coloredAngles, angleCounts = mm.giveMarkedMyocyte(
    inputs = inputs
  )
  correctColoredAngles = util.switchBRChannels(coloredAngles)
  correctColoredImg = util.switchBRChannels(coloredImg)

  ### make bar chart for content
  wtContent, ltContent, lossContent = util.assessContent(coloredImg,testImage)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('./results/fig3_BarChart.pdf',dpi=1200)
  plt.close()

  ### save files individually and arrange using inkscape
  plt.figure()
  plt.imshow(util.switchBRChannels(util.markMaskOnMyocyte(rawImg,testImage)))
  plt.gcf().savefig("./results/fig3_Raw.pdf",dpi=1200)

  plt.figure()
  plt.imshow(correctColoredAngles)
  plt.gcf().savefig("./results/fig3_ColoredAngles.pdf",dpi=1200)

  ### save histogram of angles
  du.giveAngleHistogram(angleCounts,iters,"fig3")

def HF_results(): 
  '''
  TAB results
  '''
  ### initial arguments
  filterTwoSarcSize = 25
  imgName = root + "HF_1_processed.png"
  rawImg = util.ReadImg(imgName)

  ### Read in parameters from yaml file
  inputs = mm.Inputs(
    yamlFileName='./YAML_files/HF.yml'
  )

  markedImg = mm.giveMarkedMyocyte(inputs=inputs)

  ### make bar chart for content
  wtContent, ltContent, lossContent = util.assessContent(markedImg,imgName)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### opting to make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('./results/fig4_BarChart.pdf',dpi=300)
  plt.close()
 
  switchedImg = util.switchBRChannels(markedImg)

  plt.figure()
  plt.imshow(util.switchBRChannels(util.markMaskOnMyocyte(rawImg,imgName)))
  plt.gcf().savefig("./results/fig4_Raw.pdf",dpi=300)

def MI_results(): 
  '''
  MI Results
  '''
  # filterTwoSarcSize = 25

  ### Distal, Medial, Proximal
  DImageName = root+"MI_D_76_processed.png"
  MImageName = root+"MI_M_45_processed.png"
  PImageName = root+"MI_P_16_processed.png"

  imgNames = [DImageName, MImageName, PImageName]

  ### Read in images for figure
  DImage = util.ReadImg(DImageName)
  MImage = util.ReadImg(MImageName)
  PImage = util.ReadImg(PImageName)
  images = [DImage, MImage, PImage]

  ### Define inputs for each image
  inputs_D = mm.Inputs(
    yamlFileName = './YAML_files/MI_D.yml'
  )
  inputs_M = mm.Inputs(
    yamlFileName = './YAML_files/MI_M.yml'
  )
  inputs_P = mm.Inputs(
    yamlFileName = './YAML_files/MI_P.yml'
  )

  # BE SURE TO UPDATE TESTMF WITH OPTIMIZED PARAMS
  Dimg = mm.giveMarkedMyocyte(inputs=inputs_D)
  Mimg = mm.giveMarkedMyocyte(inputs=inputs_M)
  Pimg = mm.giveMarkedMyocyte(inputs=inputs_P)

  results = [Dimg, Mimg, Pimg]
  keys = ['Distal', 'Medial', 'Proximal']
  areas = {}

  ttResults = []
  ltResults = []
  lossResults = []

  ### report responses for each case
  for i,img in enumerate(results):
    ### assess content based on cell area
    wtContent, ltContent, lossContent = util.assessContent(img,imgNames[i])
    ### store in lists
    ttResults.append(wtContent)
    ltResults.append(ltContent)
    lossResults.append(lossContent)

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  # opting to make a single bar chart
  N = 3
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, ttResults, width, color=colors[0])
  rects2 = ax.bar(indices+width, ltResults, width, color=colors[1])
  rects3 = ax.bar(indices+2*width, lossResults,width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.set_xticks(indices + width* 3/2)
  ax.set_xticklabels(keys)
  ax.legend(marks)
  plt.gcf().savefig('./results/fig5_BarChart.pdf',dpi=300)
  plt.close()

  plt.figure()
  plt.imshow(util.switchBRChannels(util.markMaskOnMyocyte(DImage,DImageName)))
  plt.gcf().savefig("./results/fig5_Raw_D.pdf",dpi=300)

  plt.figure()
  plt.imshow(util.switchBRChannels(util.markMaskOnMyocyte(MImage,MImageName)))
  plt.gcf().savefig("./results/fig5_Raw_M.pdf",dpi=300)

  plt.figure()
  plt.imshow(util.switchBRChannels(util.markMaskOnMyocyte(PImage,PImageName)))
  plt.gcf().savefig("./results/fig5_Raw_P.pdf",dpi=300)

def tissueComparison(fullAnalysis=True):
  '''
  Tissue level images for comparison between "Distal" region and "Proximal" region
  '''
  if fullAnalysis:
    fileTag = "tissueComparison_fullAnalysis"
  else:
    fileTag = "tissueComparison"

  ### Setup cases for use
  filterTwoSarcomereSize = 25
  cases = dict()
  
  cases['WTLike'] = empty()
  cases['WTLike'].loc_um = [3694,0]
  cases['WTLike'].extent_um = [300,300]
  
  cases['MILike'] = empty()
  cases['MILike'].loc_um = [2100,3020]
  cases['MILike'].extent_um = [300,300]

  tissue.SetupCase(cases['WTLike'])
  tissue.SetupCase(cases['MILike'])

  ### Store subregions for later marking
  cases['WTLike'].subregionOrig = cases['WTLike'].subregion.copy()
  cases['MILike'].subregionOrig = cases['MILike'].subregion.copy()

  ### Preprocess and analyze tissue subsections
  cases['WTLike'] = analyzeTissueCase(cases['WTLike'])
  cases['MILike'] = analyzeTissueCase(cases['MILike'])

  ### Save hits from the analysis
  displayTissueCaseHits(cases['WTLike'],tag=fileTag+'_Distal')
  displayTissueCaseHits(cases['MILike'],tag=fileTag+'_Proximal')

  if fullAnalysis:
    ### Quantify TT content per square micron
    cases['WTLike'].area = float(cases['WTLike'].extent_um[0] * cases['WTLike'].extent_um[1])
    cases['MILike'].area = float(cases['MILike'].extent_um[0] * cases['MILike'].extent_um[1])
    cases['WTLike'].TTcontent = float(np.sum(cases['WTLike'].pasted)) / cases['WTLike'].area
    cases['MILike'].TTcontent = float(np.sum(cases['MILike'].pasted)) / cases['MILike'].area
    ## Normalize TT content since it's fairly arbitrary
    cases['MILike'].TTcontent /= cases['WTLike'].TTcontent
    cases['WTLike'].TTcontent /= cases['WTLike'].TTcontent

    print ("WT TT Content:", cases['WTLike'].TTcontent)
    print ("MI TT Content:", cases['MILike'].TTcontent)

    ### Quantify TA 
    cases['WTLike'].TAcontent = float(np.sum(cases['WTLike'].TApasted))
    cases['MILike'].TAcontent = float(np.sum(cases['MILike'].TApasted))
    ### Normalize
    cases['MILike'].TAcontent /= cases['WTLike'].TAcontent
    cases['WTLike'].TAcontent /= cases['WTLike'].TAcontent

    print ("WT TA Content:", cases['WTLike'].TAcontent)
    print ("MI TA Content:", cases['MILike'].TAcontent)


  ### Make Bar Chart of TT content
  #width = 0.75
  #N = 2
  #indices = np.arange(N) + width
  #fig,ax = plt.subplots()
  #rects1 = ax.bar(indices[0], cases['WTLike'].TTcontent, width, color='blue',align='center')
  #rects2 = ax.bar(indices[1], cases['MILike'].TTcontent, width, color='red',align='center')
  #ax.set_ylabel('Normalized TT Content to WT',fontsize=24)
  #plt.sca(ax)
  #plt.xticks(indices,['Conserved','Perturbed'],fontsize=24)
  #ax.set_ylim([0,1.2])
  #plt.gcf().savefig(fileTag+'_TTcontent.pdf',dpi=300)

  ### Save enhanced original images for figure 
  plt.figure()
  plt.imshow(cases['WTLike'].displayImg,cmap='gray',vmin=0,vmax=255)
  plt.axis('off')
  plt.gcf().savefig(fileTag+'_Distal_enhancedImg.pdf',dpi=600)

  plt.figure()
  plt.imshow(cases['MILike'].displayImg,cmap='gray',vmin=0,vmax=255)
  plt.axis('off')
  plt.gcf().savefig(fileTag+'_Proximal_enhancedImg.pdf',dpi=600)

  ### Find angle counts for each rotation
  #cases['WTLike'].results.stackedAngles = cases['WTLike'].results.stackedAngles[np.where(
  #                                        cases['WTLike'].results.stackedAngles != -1
  #                                        )]
  #cases['MILike'].results.stackedAngles = cases['MILike'].results.stackedAngles[np.where(
  #                                        cases['MILike'].results.stackedAngles != -1
  #                                        )]


  ### Write stacked angles histogram
  #giveAngleHistogram(cases['WTLike'].results.stackedAngles,cases['WTLike'].iters,fileTag+"_Distal")
  #giveAngleHistogram(cases['MILike'].results.stackedAngles,cases['MILike'].iters,fileTag+"_Proximal")

def figAngleHeterogeneity():
  '''
  Figure to showcase the heterogeneity of striation angle present within the tissue
    sample
  NOTE: Not used in paper
  '''
  ### Setup case for use
  filterTwoSarcomereSize = 25
  case = empty()
  case.loc_um = [2800,3250]
  case.extent_um = [300,300]

  tissue.SetupCase(case)

  ### Store subregions for later marking
  case.subregionOrig = case.subregion.copy()

  ### Preprocess and analyze tissue subsections
  case = analyzeTissueCase(case)

  ### Save hits from the analysis
  displayTissueCaseHits(case,tag='figHeterogeneousAngles')

  ### Quantify TT content per square micron
  case.area = float(case.extent_um[0] * case.extent_um[1])
  case.TTcontent = float(np.sum(case.pasted)) / case.area
  ## Normalize TT content since it's fairly arbitrary
  case.TTcontent /= case.TTcontent

  plt.figure()
  plt.imshow(case.results.stackedAngles)
  plt.colorbar()
  plt.show()

  print (case.results.stackedAngles)

def full_ROC():
  '''
  Routine to generate the necessary ROC figures based on hand-annotated images
  '''

  root = "./myoimages/"

  imgNames = {'HF':"HF_annotation_testImg.png",
              'MI':"MI_annotation_testImg.png",
              'Control':"Sham_annotation_testImg.png"
              }

  # images that have hand annotation marked
  annotatedImgNames = {'HF':'HF_annotation_trueImg.png',
                       'Control':'Sham_annotation_trueImg.png',
                       'MI':'MI_annotation_trueImg.png'
                       }

  for key,imgName in imgNames.iteritems():
      print (imgName)
      ### setup dataset
      dataSet = optimizer.DataSet(
                  root = root,
                  filter1TestName = root + imgName,
                  filter1TestRegion=None,
                  filter1PositiveTest = root + annotatedImgNames[key],
                  pasteFilters=True
                  )

      ### run func that writes scores to hdf5 file
      myocyteROC(dataSet,key,threshes = np.linspace(0.05,0.7,30))
  
  ### read data from hdf5 files
  # make big dictionary
  bigData = {}
  bigData['MI'] = {}
  bigData['HF'] = {}
  bigData['Control'] = {}
  ### read in data from each myocyte
  for key,nestDict in bigData.iteritems():
    nestDict['WT'] = pd.read_hdf(key+"_WT.h5",'table')
    nestDict['LT'] = pd.read_hdf(key+"_LT.h5",'table')
    nestDict['Loss'] = pd.read_hdf(key+"_Loss.h5",'table')

  ### Go through and normalize all false positives and true positives
  for key,nestDict in bigData.iteritems():
    for key2,nestDict2 in nestDict.iteritems():
      #print key
      #print key2
      nestDict2['filter1PS'] /= np.max(nestDict2['filter1PS'])
      nestDict2['filter1NS'] /= np.max(nestDict2['filter1NS'])

  ### Figure generation
  plt.rcParams.update(plt.rcParamsDefault)
  f, axs = plt.subplots(3,2,figsize=[7,12])
  plt.subplots_adjust(wspace=0.5,bottom=0.05,top=0.95,hspace=0.25)
  locDict = {'Control':0,'MI':1,'HF':2}
  for key,loc in locDict.iteritems():
    ### writing detection rate fig
    axs[loc,0].scatter(bigData[key]['WT']['filter1Thresh'], 
                     bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,0].scatter(bigData[key]['LT']['filter1Thresh'], 
                     bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,0].scatter(bigData[key]['Loss']['filter1Thresh'],
                     bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    axs[loc,0].set_title(key+" Detection Rate",size=12)
    axs[loc,0].set_xlabel('Threshold')
    axs[loc,0].set_ylabel('Detection Rate')
    axs[loc,0].set_ylim([0,1])
    axs[loc,0].set_xlim(xmin=0)
  
    ### writing ROC fig
    axs[loc,1].set_title(key+" ROC",size=12)
    axs[loc,1].scatter(bigData[key]['WT']['filter1NS'], 
                       bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,1].scatter(bigData[key]['LT']['filter1NS'],    
                       bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,1].scatter(bigData[key]['Loss']['filter1NS'],    
                       bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    ### giving 50% line
    vert = np.linspace(0,1,10)
    axs[loc,1].plot(vert,vert,'k--')

    axs[loc,1].set_xlim([0,1])
    axs[loc,1].set_ylim([0,1])
    axs[loc,1].set_xlabel('False Positive Rate (Normalized)')
    axs[loc,1].set_ylabel('True Positive Rate (Normalized)')

  plt.gcf().savefig('figS1.pdf',dpi=300)

def tissueBloodVessel():
  '''
  Routine to generate the figure showcasing heterogeneity of striation angle
    in the tissue sample
  '''
  fileTag = 'tissueBloodVessel'

  ### setup case
  case = empty()
  case.loc_um = [2477,179]
  case.extent_um = [350,350]
  case.orig = tissue.Setup()
  case.subregion = tissue.get_fiji(case.orig,case.loc_um,case.extent_um)
  #case = tissue.SetupTest()

  ### Store subregions for later marking
  case.subregionOrig = case.subregion.copy()

  ### Preprocess and analyze the case
  case = analyzeTissueCase(case)

  ### display the hits of the case
  displayTissueCaseHits(case,fileTag)

  ### Save enhanced original images for figure
  plt.figure()
  plt.imshow(case.displayImg,cmap='gray',vmin=0,vmax=255)
  plt.gcf().savefig(fileTag+'_enhancedImg.pdf',dpi=300)

def algComparison():
  '''
  Routine to compare the GPU and CPU implementation of code
  '''

  fileTag = "algComparison"

  ### setup case
  caseGPU = empty()
  caseGPU.loc_um = [2477,179]
  caseGPU.extent_um = [350,350]
  caseGPU.orig = tissue.Setup()
  caseGPU.subregion = tissue.get_fiji(caseGPU.orig,caseGPU.loc_um,caseGPU.extent_um)

  ### Store subregions for later marking
  caseGPU.subregionOrig = caseGPU.subregion.copy()

  ### Preprocess 
  caseGPU = preprocessTissueCase(caseGPU)

  ### make necessary copies for CPU case
  caseCPU = empty()
  caseCPU.loc_um = [2477,179]
  caseCPU.extent_um = [350,350]
  caseCPU.orig = caseGPU.orig.copy()
  caseCPU.degreesOffCenter = caseGPU.degreesOffCenter
  caseCPU.subregionOrig = caseGPU.subregionOrig.copy()
  caseCPU.subregion = caseGPU.subregion.copy()
  caseCPU.displayImg = caseGPU.displayImg.copy()
  
  ### analyze the case
  caseGPU = analyzeTissueCase(caseGPU,preprocess=False)

  ### display the hits of the case
  displayTissueCaseHits(caseGPU,fileTag+"_GPU")

  ### store results
  GPUpastedHits = caseGPU.pasted

  ### Preprocess and analyze the case
  caseCPU = analyzeTissueCase(caseCPU,preprocess=False,useGPU=False)

  ### display the hits of the case
  displayTissueCaseHits(caseCPU,fileTag+"_CPU")

  ### store results
  CPUpastedHits = caseCPU.pasted

  ### save original enhanced image for comparison
  plt.figure()
  plt.imshow(caseGPU.displayImg,cmap='gray',vmin=0,vmax=255)
  plt.gcf().savefig(fileTag+'_enhancedImg.pdf',dpi=600)
  
  ### do comparison between GPU and CPU results
  comparison = np.abs(GPUpastedHits - CPUpastedHits).astype(np.float32)
  comparison /= np.max(comparison)
  plt.figure()
  plt.imshow(comparison,cmap='gray')
  plt.colorbar()
  plt.gcf().savefig(fileTag+'_comparison.pdf',dpi=600)

def YAML_example():
  '''
  Routine to generate the example YAML output in the supplement
  '''
  raise RuntimeError('Function is deprecated. This is now generated by calling matchedmyo.py with' 
                     +'the example yaml file in the manuscript')
  detect.updatedSimpleYaml("ex.yml")

def saveWorkflowFig():
  '''
  Function that will save the images used for the workflow figure in the paper.
  Note: This is slightly subject to how one preprocesses the MI_D_73.png image.
        A slight change in angle or what subsection was selected in the preprocessing
        could slightly change how the images appear.
  '''

  imgName = "./myoimages/MI_D_73_processed.png" 
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]

  inputs = mm.Inputs(
    yamlFileName='./YAML_files/workflow.yml'
  )
  
  colorImg,colorAngles,angleCounts = mm.giveMarkedMyocyte(
    inputs=inputs
  )

  ### save the correlation planes
  lossFilter = util.LoadFilter("./myoimages/LossFilter.png")
  ltFilter = util.LoadFilter("./myoimages/LongitudinalFilter.png")
  wtFilter = util.LoadFilter("./myoimages/newSimpleWTFilter.png")

  origImg = util.ReadImg(imgName,cvtColor=False)
  origImg = cv2.cvtColor(origImg,cv2.COLOR_BGR2GRAY)
  lossCorr = mF.matchedFilter(origImg, lossFilter,demean=False)
  ltCorr = mF.matchedFilter(origImg, ltFilter,demean=False)
  wtCorr = mF.matchedFilter(origImg, wtFilter,demean=False)

  cropImgs = False
  if cropImgs:
    angle_output = util.ReadImg("WorkflowFig_angles_output.png",cvtColor=False)
    output = util.ReadImg("WorkflowFig_output.png",cvtColor=False)
    imgs = {"WorkflowFig_angles_output.png":angle_output, 
            "WorkflowFig_output.png":output, 
            "WorkflowFig_orig.png":origImg}

    left = 204; right = 304; top = 74; bottom = 151
    for name,img in imgs.iteritems():
      if cropImgs:
        holder = np.zeros((bottom-top,right-left,3),dtype=np.uint8)
        for channel in range(3):
          ### crop images
          holder[:,:,channel] = img[top:bottom,left:right,channel]
      cv2.imwrite(name,holder)
    lossCorr = lossCorr[top:bottom,left:right]
    ltCorr = ltCorr[top:bottom,left:right]
    wtCorr = wtCorr[top:bottom,left:right]

  plt.figure()
  plt.imshow(lossCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_lossCorr.pdf")

  plt.figure()
  plt.imshow(ltCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_ltCorr.pdf")

  plt.figure()
  plt.imshow(wtCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_wtCorr.pdf")

  ### Assess content
  wtC,ltC,ldC = util.assessContent(colorImg,imgName=imgName)

  ### make a bar chart using routine
  content = [wtC,ltC,ldC]
  contentDict = {imgName:content}
  du.giveBarChartfromDict(contentDict,"WorkflowFig_Content")

  ### Make a histogram for the angles
  du.giveAngleHistogram(angleCounts,iters,"WorkflowFig")

def test3DSimulatedData(LT_probability = [0., .15, .3], 
                        TA_probability = [0., .15, .3],
                        noiseAmplitude = 0.,
                        numReplicates = 20):
  '''This function tests out the implementation of the 3D data simulation and detection routines.
  This is carried out by generating the simulated data based on user specified ratios of LT, TT,
  and TA unit cells. We then analyze this using the 3D filtering routines. Then compare the results
  compared to what we would expect based on the specified probability distributions.

  Inputs:
    LT_probability -> List of floats. Probability of inserting a LT unit cell into the simulated data.
    TA_probability -> List of floats. Probability of inserting a TA unit cell into the simulated data.
                        NOTE: The probability of finding a TT unit cell is
                          1 - LT_probability - TA_probability
    noiseAmplitude -> float. Amplitude of the multiplicative Gaussian white noise in the simulated data
    numReplicates -> int. Number of cells generated for each combination of probabilities

  Outputs:
    None
  '''

  ### Define parameters for the simulation of the cell
  ## Define scope resolutions for generating the filters and the cell. This is in x, y, and z resolutions
  scopeResolutions = [10,10,5] #[vx / um]
  ## x, y, and z Dimensions of the simulated cell [microns]
  cellDimensions = [20, 20, 30]
  ## Give names for your filters. NOTE: These are hardcoded in the filter generation routines in util.py
  ttName = './myoimages/TT_3D.tif'
  ttPunishName = './myoimages/TT_Punishment_3D.tif'
  ltName = './myoimages/LT_3D.tif'
  taName = './myoimages/TA_3D.tif'

  ### Loop through and generate all of the necessary simulated cells and classification results
  storage = {}
  for LTp in LT_probability:
    for TAp in TA_probability:
      ## Create storage entry in dictionary
      probs = str(LTp)+'_'+str(TAp)
      storage[probs] = {
        'TA': [], 
        'LT': [],
        'TT': []
      }
      for num in range(numReplicates):
        ## Define test file name
        testName = "./myoimages/3DTestData_{}LT_{}TA.tif".format(
          str(LTp).replace('.',''),
          str(TAp).replace('.','')
        )

        ## Simulate the small 3D cell
        util.generateSimulated3DCell(LT_probability=LTp,
                                     TA_probability=TAp,
                                     noiseAmplitude=noiseAmplitude,
                                     scopeResolutions=scopeResolutions,
                                     cellDimensions=cellDimensions,
                                     fileName=testName
                                     )
        
        ## Form inputs
        inputs = mm.Inputs(
          imageName = testName,
          scopeResolutions = scopeResolutions,
          preprocess=False
        )
        inputs.dic['iters'] = [[0,0,0]]
        inputs.dic['preprocess'] = False
        inputs.dic['dimensions'] = 3
        # Update parameter dicitonaries to now contain 3D parameters, not 2D
        inputs.updateParamDicts()

        ## Classify the simulated cell
        myResults = mm.give3DMarkedMyocyte(
          inputs = inputs
        )

        cellVolume = float(np.prod(inputs.imgOrig.shape))

        storage[probs]['TA'].append(float(np.count_nonzero(myResults.markedImage[..., 2] == 255)) / cellVolume)
        storage[probs]['LT'].append(float(np.count_nonzero(myResults.markedImage[..., 1] == 255)) / cellVolume)
        storage[probs]['TT'].append(float(np.count_nonzero(myResults.markedImage[..., 0] == 255)) / cellVolume)

  with open('./results/test3DSimulatedData.pkl', 'wb') as fil:
    pkl.dump(storage, fil)

  for probs, contentDict in storage.iteritems():
    print (probs+':')
    print ('\tTA Content '+str(np.mean(contentDict['TA']))[:5]+' +- '+str(np.std(contentDict['TA']))[:5])
    print ('\tLT Content '+str(np.mean(contentDict['LT']))[:5]+' +- '+str(np.std(contentDict['LT']))[:5])
    print ('\tTT Content '+str(np.mean(contentDict['TT']))[:5]+' +- '+str(np.std(contentDict['TT']))[:5])


###################################################################################################
###################################################################################################
###################################################################################################
###
### Utility Functions Specifically for Figure Generation
###
###################################################################################################
###################################################################################################
###################################################################################################

def preprocessTissueCase(case):
  ### Preprocess subregions
  case.subregion, case.degreesOffCenter = pp.reorient(
          case.subregion
          )

  ### save image for display later
  ## I'm considering writing a better routine to enhance the image for figure quality but not necessarily for algorithm quality
  brightnessDamper = 0.6
  case.displayImg = case.subregion.copy().astype(np.float32) / float(np.max(case.subregion))
  case.displayImg *= brightnessDamper * 255.
  case.displayImg = case.displayImg.astype(np.uint8)

  return case

def setupAnnotatedImage(annotatedName, baseImageName):
  '''
  Function to be used in conjunction with Myocyte().
  Uses the util.markPastedFilters() function to paste filters onto the annotated image.
  This is so we don't have to generate a new annotated image everytime we 
  change filter sizes.
  '''
  ### Read in images
  #baseImage = util.ReadImg(baseImageName,cvtColor=False)
  markedImage = util.ReadImg(annotatedName, cvtColor=False)
  
  ### Divide up channels of markedImage to represent hits
  wtHits, ltHits = markedImage[:,:,0],markedImage[:,:,1]
  wtHits[wtHits > 0] = 255
  ltHits[ltHits > 0] = 255
  # loss is already adequately marked so we don't want it ran through the routine
  lossHits = np.zeros_like(wtHits)
  coloredImage = util.markPastedFilters(lossHits,ltHits,wtHits,markedImage)
  # add back in the loss hits
  coloredImage[:,:,2] = markedImage[:,:,2]  

  ### Save image to run with optimizer routines
  newName = annotatedName[:-4]+"_pasted"+annotatedName[-4:]
  cv2.imwrite(newName,coloredImage)

  return newName


###################################################################################################
###################################################################################################
###################################################################################################
###
###  Plotting/Display Functions
###
###################################################################################################
###################################################################################################
###################################################################################################

def displayTissueCaseHits(case,
                          tag,
                          displayTT=True,
                          displayTA=True):
  '''
  Displays the 'hits' returned from analyzeTissueCase() function
  '''
  ### Convert subregion back into cv2 readable format
  case.subregion = np.asarray(case.subregion * 255.,dtype=np.uint8)

  ### Mark where the filter responded and display on the images
  ## find filter dimensions
  TTy,TTx = util.measureFilterDimensions(case.inputs.mfOrig)

  ## mark unit cells on the image where the filters responded
  case.pasted = painter.doLabel(case.results,
                                cellDimensions=[TTx,TTy],
                                thresh=case.params['snrThresh'])

  ## convert pasted filter image to cv2 friendly format and normalize original subregion
  case.pasted = np.asarray(case.pasted 
                           / np.max(case.pasted) 
                           * 255.,
                           dtype=np.uint8)

  ## rotate images back to the original orientation
  case.pasted = imutils.rotate(case.pasted,case.degreesOffCenter)
  case.displayImg = imutils.rotate(case.displayImg,case.degreesOffCenter)

  debug = False
  if debug:
    plt.figure()
    plt.imshow(case.displayImg,cmap='gray')
    plt.show()

  ## cut image back down to original size to get rid of borders
  imgDims = np.shape(case.subregionOrig)
  origY,origX = float(imgDims[0]), float(imgDims[1])
  newImgDims = np.shape(case.displayImg)
  newY,newX = float(newImgDims[0]),float(newImgDims[1])
  padY,padX = int((newY - origY)/2.), int((newX - origX)/2.)
  case.pasted = case.pasted[padY:-padY,
                            padX:-padX]
  case.displayImg = case.displayImg[padY:-padY,
                                    padX:-padX]

  ### Create colored image for display
  coloredImage = np.asarray((case.displayImg.copy(),
                             case.displayImg.copy(),
                             case.displayImg.copy()))
  coloredImage = np.rollaxis(coloredImage,0,start=3)
  TTchannel = 2

  ### Mark channel hits on image
  if displayTT:
    coloredImage[case.pasted != 0,TTchannel] = 255

  ### Do the same thing for the TA case
  if displayTA:
    TAy,TAx = util.measureFilterDimensions(case.TAinputs.mfOrig)
    case.TApasted = painter.doLabel(case.TAresults,
                                    cellDimensions=[TAx,TAy],
                                    thresh=case.TAparams['snrThresh'])
    case.TApasted = np.asarray(case.TApasted
                               / np.max(case.TApasted)
                               * 255.,
                               dtype=np.uint8)
    case.TApasted = imutils.rotate(case.TApasted,case.degreesOffCenter)
    case.TApasted = case.TApasted[padY:-padY,
                                  padX:-padX]
    TAchannel = 0
    coloredImage[case.TApasted != 0,TAchannel] = 255

  ### Plot figure and save
  plt.figure()
  plt.imshow(coloredImage,vmin=0,vmax=255)
  plt.gcf().savefig(tag+"_hits.pdf",dpi=600)


###################################################################################################
###################################################################################################
###################################################################################################
###
### Analysis Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def analyzeTissueCase(case,
                      preprocess=True,
                      useGPU=True,
                      analyzeTA=True):
  '''
  Refactored method to analyze tissue cases 
  '''
  if preprocess:
    case = preprocessTissueCase(case)

  ### Setup Filters
  root = "./myoimages/"
  ttFilterName = root+"newSimpleWTFilter.png"
  ttPunishmentFilterName = root+"newSimpleWTPunishmentFilter.png"
  case.iters = [-25,-20,-15,-10-5,0,5,10,15,20,25]
  returnAngles = True
  ttFilter = util.LoadFilter(ttFilterName)
  ttPunishmentFilter = util.LoadFilter(ttPunishmentFilterName)

  ### Setup parameter dictionaries
  case.params = optimizer.ParamDict(typeDict="TT")
  case.params['covarianceMatrix'] = np.ones_like(case.subregion)
  case.params['mfPunishment'] = ttPunishmentFilter
  case.params['useGPU'] = useGPU

  ### Setup input classes
  case.inputs = empty()
  case.inputs.imgOrig = case.subregion.astype(np.float32) / float(np.max(case.subregion))
  case.inputs.mfOrig = ttFilter

  ### Perform filtering for TT detection
  case.results = bD.DetectFilter(case.inputs,
                                 case.params,
                                 case.iters
                                 )

  ### Modify case to perform TA detection
  if analyzeTA:
    case.TAinputs = empty()
    case.TAinputs.imgOrig = case.subregion
    case.TAinputs.displayImg = case.displayImg
    lossFilterName = root+"LossFilter.png"
    case.TAIters = [-45,0]
    lossFilter = util.LoadFilter(lossFilterName)
    case.TAinputs.mfOrig = lossFilter
    case.TAparams = optimizer.ParamDict(typeDict='TA')

    ### Perform filtering for TA detection
    case.TAresults = bD.DetectFilter(case.TAinputs,
                                     case.TAparams,
                                     case.TAIters)

  return case

def analyzeAllMyo(root="/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"):
  '''
  Function to iterate through a directory containing images that have already
  been preprocessed by preprocessing.py
  This directory can contain masks but it is not necessary
  '''
  ### instantiate dicitionary to hold content values
  Sham = {}; MI_D = {}; MI_M = {}; MI_P = {}; HF = {}

  for name in os.listdir(root):
    if "mask" in name:
      continue
    print (name)
    iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25]

    ### Form inputs
    inputs = mm.Inputs(
      imageName=name
    )
    inputs.dic['outputParams'] = {
      'fileRoot':name[:-4],
      'fileType':'pdf',
      'dpi':1200
    }

    ### iterate through names and mark the images
    markedMyocyte,_,angleCounts = mm.giveMarkedMyocyte(
      inputs = inputs
    )

    ### save raw image with ROI marked
    cImg = util.ReadImg(root+name)
    cImg = util.markMaskOnMyocyte(cImg,root+name)
    plt.figure()
    plt.imshow(util.switchBRChannels(cImg))
    plt.gcf().savefig(name[:-4]+'_markedMask.pdf')
    plt.close()

    ### hacky way to get percent of hits within range of 5 degrees from minor axis
    idxs = [4,5,6]
    totalHits = len(angleCounts)
    angleCountsNP = np.asarray(angleCounts)
    hitsInRange =   np.count_nonzero(np.equal(angleCounts, iters[idxs[0]])) \
                  + np.count_nonzero(np.equal(angleCounts, iters[idxs[1]])) \
                  + np.count_nonzero(np.equal(angleCounts, iters[idxs[2]])) 
    print ("Percentage of WT hits within 5 degrees of minor axis:", float(hitsInRange)/float(totalHits) * 100.)

    ### assess content
    wtC, ltC, lossC = util.assessContent(markedMyocyte,imgName=root+name)
    content = np.asarray([wtC, ltC, lossC],dtype=float)

    ### store content in respective dictionary
    if 'Sham' in name:
      Sham[name] = content
      Sham[name+'_angles'] = angleCounts
    elif 'HF' in name:
      HF[name] = content
      HF[name+'_angles'] = angleCounts
    elif 'MI' in name:
      if '_D' in name:
        MI_D[name] = content
        MI_D[name+'_angles'] = angleCounts
      elif '_M' in name:
        MI_M[name] = content
        MI_M[name+'_angles'] = angleCounts
      elif '_P' in name:
        MI_P[name] = content
        MI_P[name+'_angles'] = angleCounts

    ### make angle histogram for the data
    du.giveAngleHistogram(angleCounts,iters,tag=name[:-4])

  ### use function to construct and write bar charts for each content dictionary
  du.giveBarChartfromDict(Sham,'Sham')
  du.giveBarChartfromDict(HF,'HF')
  du.giveMIBarChart(MI_D,MI_M,MI_P)
  du.giveAvgStdofDicts(Sham,HF,MI_D,MI_M,MI_P)

def Myocyte():
  '''This function defines the dataset for the myocyte optimization routines
  '''

  # where to look for images
  root = "myoimages/"

  filter1TestName = root + "MI_annotation_testImg.png"
  filter1PositiveTest = root + "MI_annotation_trueImg.png"

  dataSet = optimizer.DataSet(
      root = root,
      filter1TestName = filter1TestName,
      filter1TestRegion = None,
      filter1PositiveTest = filter1PositiveTest,
      filter1PositiveChannel= 0,  # blue, WT 
      filter1Label = "TT",
      filter1Name = root+'WTFilter.png',          
      filter1Thresh=0.06, 

      filter2TestName = filter1TestName,
      filter2TestRegion = None,
      filter2PositiveTest = filter1PositiveTest,
      filter2PositiveChannel= 1,  # green, longi
      filter2Label = "LT",
      filter2Name = root+'newLTfilter.png',        
      filter2Thresh=0.38 
  )


  # flag to paste filters on the myocyte to smooth out results
  dataSet.pasteFilters = True

  return dataSet

def rocData(): 
  dataSet = Myocyte() 

  # rotation angles
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]

  root = "./myoimages/"

  # flag to turn on the pasting of unit cell on each hit
  dataSet.pasteFilters = True

  ## Testing TT first 
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'newSimpleWTFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='TT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png")
  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(0.1,0.45, 25),
        iters=iters,
        )

  ## Testing LT now
  dataSet.filter1PositiveChannel=1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  #dataSet.filter1Name = root+'newLTfilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='LT')  

  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(0.01, 0.4, 25),
        iters=iters
      )

  ## Testing Loss
  dataSet.filter1PositiveChannel = 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='TA')
  lossIters = [0,45]

  optimizer.GenFigROC_TruePos_FalsePos(
         dataSet,
         paramDict,
         filter1Label = dataSet.filter1Label,
         f1ts = np.linspace(0.005,0.1,25),
         iters=lossIters,
       )

def myocyteROC(data, myoName,
               threshes = np.linspace(5,30,10),
               iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
               ):
  '''Function to calculate data for a full ROC for a given myocyte and return
  scores for each filter at given thresholds
  '''
  root = "./myoimages/"

  ### WT
  # setup WT data in class structure
  data.filter1PositiveChannel= 0
  data.filter1Label = "TT"
  data.filter1Name = root + 'newSimpleWTFilter.png'
  optimizer.SetupTests(data,meanFilter=True)
  WTparams = optimizer.ParamDict(typeDict='TT')
  WTparams['covarianceMatrix'] = np.ones_like(data.filter1TestData)
  WTparams['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png")


  # write filter performance data for WT into hdf5 file
  optimizer.Assess_Single(data, 
                          WTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_WT.h5",
                          display=False,
                          iters=iters)
  
  ### LT
  # setup LT data
  data.filter1PositiveChannel=1
  data.filter1Label = "LT"
  data.filter1Name = root+'LongitudinalFilter.png'
  optimizer.SetupTests(data,meanFilter=True)
  data.meanFilter = True
  LTparams = optimizer.ParamDict(typeDict='LT')

  # write filter performance data for LT into hdf5 file
  optimizer.Assess_Single(data, 
                          LTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_LT.h5",
                          display=False,
                          iters=iters)

  ### Loss  
  # setup Loss data
  data.filter1PositiveChannel = 2
  data.filter1Label = "Loss"
  data.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(data,meanFilter=True)
  Lossparams = optimizer.ParamDict(typeDict='TA')
  LossIters = [0,45]

  # write filter performance data for Loss into hdf5 file
  optimizer.Assess_Single(data, 
                          Lossparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_Loss.h5",
                          display=False,
                          iters=LossIters)

def minDistanceROC(dataSet,paramDict,param1Range,param2Range,
                   param1="snrThresh",
                   param2="stdDevThresh",
                   FPthresh=0.1,
                   iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
                   ):
  '''
  Function that will calculate the minimum distance to the perfect detection point
  (0,1) on a ROC curve and return those parameters.
  '''
  perfectDetection = (0,1)

  distanceStorage = np.ones((len(param1Range),len(param2Range)),dtype=np.float32)
  TruePosStorage = np.ones_like(distanceStorage)
  FalsePosStorage = np.ones_like(distanceStorage)
  for i,p1 in enumerate(param1Range):
    paramDict[param1] = p1
    for j,p2 in enumerate(param2Range):
      paramDict[param2] = p2
      print ("Param 1:",p1)
      print ("Param 2:",p2)
      # having to manually assign the thresholds due to structure of TestParams function
      if param1 == "snrThresh":
        dataSet.filter1Thresh = p1
      elif param2 == "snrThresh":
        dataSet.filter1Thresh = p2
      posScore,negScore = optimizer.TestParams_Single(dataSet,paramDict,iters=iters)
      TruePosStorage[i,j] = posScore
      FalsePosStorage[i,j] = negScore
      if negScore < FPthresh:
        distanceFromPerfect = np.sqrt(((perfectDetection[0]-negScore)**2 +\
                                      (perfectDetection[1]-posScore)**2))
        distanceStorage[i,j] = distanceFromPerfect

  idx = np.unravel_index(distanceStorage.argmin(), distanceStorage.shape)
  optP1idx,optP2idx = idx[0],idx[1]
  optimumP1 = param1Range[optP1idx]
  optimumP2 = param2Range[optP2idx]
  optimumTP = TruePosStorage[optP1idx,optP2idx]
  optimumFP = FalsePosStorage[optP1idx,optP2idx]

  print ("")
  print (100*"#")
  print ("Minimum Distance to Perfect Detection:",distanceStorage.min())
  print ("True Postive Rate:",optimumTP)
  print ("False Positive Rate:",optimumFP)
  print ("Optimum",param1,"->",optimumP1)
  print ("Optimum",param2,"->",optimumP2)
  print (100*"#")
  print ("")
  return optimumP1, optimumP2, distanceStorage

def optimizeWT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'newSimpleWTFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)

  paramDict = optimizer.ParamDict(typeDict='TT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png") 
  #snrThreshRange = np.linspace(0.01, 0.15, 35)
  #gammaRange = np.linspace(4., 25., 35)
  snrThreshRange = np.linspace(.1, 0.7, 20)
  gammaRange = np.linspace(1., 4., 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,gammaRange,
                                                  param1="snrThresh",
                                                  param2="gamma", FPthresh=1.)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_WT.png")
  
def optimizeLT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  optimizer.SetupTests(dataSet)

  paramDict = optimizer.ParamDict(typeDict='LT')
  snrThreshRange = np.linspace(0.4, 0.8, 20)
  stdDevThreshRange = np.linspace(0.05, 0.4, 20)

  FPthresh = 1.

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=FPthresh)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_LT.png")

def optimizeLoss():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+'LossFilter.png'
  optimizer.SetupTests(dataSet)

  paramDict = optimizer.ParamDict(typeDict='TA')
  snrThreshRange = np.linspace(0.05,0.3, 20)
  stdDevThreshRange = np.linspace(0.05, 0.2, 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=1.)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_Loss.png")

###################################################################################################
###################################################################################################
###################################################################################################
###
###  Validation Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

###
### Function to test that the optimizer routines that assess positive and negative
### filter scores are working correctly.
###
def scoreTest():
  dataSet = Myocyte() 

  ## Testing TT first 
  dataSet.filter1PositiveChannel=0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet)
  dataSet.filter1Thresh = 5.5

  paramDict = optimizer.ParamDict(typeDict='TT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)

  filter1PS,filter1NS = optimizer.TestParams_Single(
    dataSet,
    paramDict,
    iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
    display=False)  
    #display=True)  

  print (filter1PS, filter1NS)

  val = 0.926816518557
  assert((filter1PS - val) < 1e-3), "Filter 1 Positive Score failed"
  val = 0.342082872458
  assert((filter1NS - val) < 1e-3), "Filter 1 Negative Score failed"
  print ("PASSED")


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

    ### Validation Routines
    if(arg=="-scoretest"):
      scoreTest()             
      quit()
    

    ### Figure Generation Routines

    # this function will generate input data for the current fig #3 in the paper 
    if(arg=="-WT"):               
      WT_results()
      quit()

    if(arg=="-HF"):               
      HF_results()
      quit()

    if(arg=="-MI"):               
      MI_results()
      quit()

    if(arg=="-tissueComparison"):               
      tissueComparison(fullAnalysis=True)
      quit()

    if(arg=="-figAngle"):
      figAngleHeterogeneity()
      quit()

    if(arg=="-full_ROC"):
      full_ROC()
      quit()

    if(arg=="-tissueBloodVessel"):
      tissueBloodVessel()
      quit()

    if(arg=="-algComparison"):
      algComparison()
      quit()

    if(arg=="-yaml"):
      YAML_example()
      quit()

    if(arg=='-test3DSimulatedData'):
      test3DSimulatedData()
      quit()

    # generates all figs
    if(arg=="-allFigs"):
      WT_results()     
      HF_results()     
      MI_results()
      tissueComparison()     
      full_ROC()
      tissueBloodVessel()
      algComparison()
      YAML_example()
      quit()

    if(arg=="-workflowFig"):
      saveWorkflowFig()
      quit()

    ### Testing/Optimization Routines
    if(arg=="-roc"): 
      rocData()
      quit()

    if(arg=="-optimizeWT"):
      optimizeWT()
      quit()

    if(arg=="-optimizeLT"):
      optimizeLT()
      quit()

    if(arg=="-optimizeLoss"):
      optimizeLoss()
      quit()
	   
    if(arg=="-test"):
      mm.giveMarkedMyocyte(      
        ttFilterName=sys.argv[i+1],
        ltFilterName=sys.argv[i+2],
        testImage=sys.argv[i+3],           
        ttThresh=np.float(sys.argv[i+4]),           
        ltThresh=np.float(sys.argv[i+5]),
        gamma=np.float(sys.argv[i+6]),
        ImgTwoSarcSize=(sys.argv[i+7]),
	tag = tag,
	writeImage = True)            
      quit()

    if(arg=="-testMyocyte"):
      testImage = sys.argv[i+1]
      mm.giveMarkedMyocyte(testImage=testImage,
                        tag="Testing",
                        writeImage=True)
      quit()

    if(arg=="-analyzeAllMyo"):
      analyzeAllMyo()
      quit()

    if(arg=="-testTissue"):
      name = "testingNotchFilter.png"
      mm.giveMarkedMyocyte(testImage=name,
                        tag="TestingNotchedFilter",
                        iters=[-5,0,5],
                        returnAngles=False,
                        writeImage=True,
                        useGPU=True)
      quit()

    ### Additional Arguments
    if(arg=="-tag"):
      tag = sys.argv[i+1]

    if(arg=="-noPrint"):
      sys.stdout = open(os.devnull, 'w')

    if(arg=="-shaveFig"):
      fileName = sys.argv[i+1]
      try:
        padY = int(sys.argv[i+2])
        padX = int(sys.argv[i+3])
        whiteSpace = int(sys.argv[i+4])
      except:
        padY = None
        padX = None
        whiteSpace = None
      du.shaveFig(fileName,padY=padY,padX=padX,whiteSpace=whiteSpace)
      quit()

  raise RuntimeError("Arguments not understood")

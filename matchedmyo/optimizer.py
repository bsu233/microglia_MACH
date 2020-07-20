from __future__ import print_function
"""
Purpose of this program is to optimize the threshold parameters 
to cleanly pick out data features

"""
import matplotlib
#matplotlib.use('Agg')
import cv2
import sys
import pandas as pd
import bankDetect as bD
import numpy as np
import matplotlib.pylab as plt
import util
import painter

class empty():pass

##
## dataset for param optimziation 
## This class is intended to contain all the necessary information to optmize detection parameters. 
## They will generally be specific to each type of data set. The definition below defaults to 
## parameters used for a silica imaging project, but they should be updated as appropriate for new
## data sets 
## 
root = "pnpimages/"
class DataSet:
  def __init__(self, 
    root = None,
    # filter1
    filter1TestName = root + 'clahe_Best.jpg',
    filter1TestRegion = [340,440,400,500],  # None 
    filter1Label= 'fused',
    filter1Name = root+'fusedCellTEM.png',
    filter1PositiveTest = root+"fusedMarked.png",
    filter1PositiveChannel= -1, # [012] for R, G B
    filter1Thresh=1000.,
    # filter2
    filter2TestName = root + 'clahe_Best.jpg',
    filter2TestRegion = [250,350,50,150],   # None 
    filter2Label= 'bulk',   
    filter2Name = root+'bulkCellTEM.png',
    filter2PositiveTest = root+"bulkMarked.png",
    filter2PositiveChannel= -1, # [012] for R, G B
    filter2Thresh=1050.,
    sigma_n = 1.,
    penaltyscale = 1., # TODO remove me 
    useFilterInv = False,  # need to antiquate this   
    pasteFilters = False
    ): 

    self.root = root 
    self.filter1TestName =filter1TestName #  root + 'clahe_Best.jpg'
    self.filter1TestRegion =filter1TestRegion #  [340,440,400,500]
    self.filter1Label =filter1Label #  root+'fusedCellTEM.png'
    self.filter1Name =filter1Name #  root+'fusedCellTEM.png'
    self.filter1PositiveTest =filter1PositiveTest #  root+"fusedMarked.png"
    self.filter1PositiveChannel = filter1PositiveChannel
    self.filter1Thresh=filter1Thresh #h1000.

    ## Plan to make this optional
    self.filter2TestName =filter2TestName #  root + 'clahe_Best.jpg'
    self.filter2TestRegion =filter2TestRegion #  [250,350,50,150]
    self.filter2Label =filter2Label #  root+'fusedCellTEM.png'
    self.filter2Name =filter2Name #  root+'bulkCellTEM.png'
    self.filter2PositiveTest =filter2PositiveTest #  root+"bulkMarked.png"
    self.filter2PositiveChannel = filter2PositiveChannel
    self.filter2Thresh=filter2Thresh #h1050.

    self.sigma_n = sigma_n
    self.useFilterInv = useFilterInv 
    self.penaltyscale = penaltyscale
    self.pasteFilters = pasteFilters

##
## Has some logic for processing cropped data and data with multiple channels
## Todo: prolly should load filters here too and remove the IO from bankDetect
##
def SetupTests(dataSet,meanFilter=False):
  
  #filter1/, designate channel
  def LoadFilterData(testDataName, subsection, # may be None
                     truthName, truthChannel): 
    ## load data against which filters are tested
    testData = cv2.imread(testDataName)
    testData = cv2.cvtColor(testData, cv2.COLOR_BGR2GRAY)
    testData = util.renorm(np.array(testData,dtype=float),scale=1.)

    # crop 
    if isinstance(subsection, (list, tuple, np.ndarray)):
      testData = testData[
        subsection[0]:subsection[1],subsection[2]:subsection[3]]

    ## positive test
    # use a specific channel 
    truthData = cv2.imread(truthName) 
    if truthChannel>-1:
      truthData = truthData[:,:, truthChannel ] 
    else:
      truthData = cv2.cvtColor(truthData, cv2.COLOR_BGR2GRAY)
    truthData= np.array(truthData> 0, dtype=np.float)
    truthData = util.renorm(truthData,scale=1.)

    msg =   "%s != %s"%(np.shape(truthData), np.shape(testData))
    assert( np.prod(np.shape(truthData)) == np.prod(np.shape(testData))), msg

    return testData, truthData

  # filter1
  dataSet.filter1TestData, dataSet.filter1PositiveData = LoadFilterData(
    dataSet.filter1TestName,
    dataSet.filter1TestRegion,
    dataSet.filter1PositiveTest,
    dataSet.filter1PositiveChannel
    )
  # filter2
  dataSet.filter2TestData, dataSet.filter2PositiveData = LoadFilterData(
    dataSet.filter2TestName,
    dataSet.filter2TestRegion,
    dataSet.filter2PositiveTest,
    dataSet.filter2PositiveChannel
    )
        # load fused filter

  # load filters
  #print dataSet.filter1Name
  if meanFilter:
    dataSet.filter1Data = util.LoadFilter(dataSet.filter1Name)
  else:
    filter1Filter = cv2.imread(dataSet.filter1Name)
    dataSet.filter1Data   = cv2.cvtColor(filter1Filter, cv2.COLOR_BGR2GRAY)
    dataSet.filter1Data = util.renorm(np.array(dataSet.filter1Data,dtype=float),scale=1.)
  dataSet.filter1y,dataSet.filter1x = util.measureFilterDimensions(dataSet.filter1Data)

  if meanFilter:
    dataSet.filter2Data = util.LoadFilter(dataSet.filter2Name)
  else:
    filter2Filter = cv2.imread(dataSet.filter2Name)
    dataSet.filter2Data   = cv2.cvtColor(filter2Filter, cv2.COLOR_BGR2GRAY)
    dataSet.filter2Data = util.renorm(np.array(dataSet.filter2Data,dtype=float),scale=1.)
  dataSet.filter2y,dataSet.filter2x = util.measureFilterDimensions(dataSet.filter2Data)

def ParamDict(typeDict=''):
  paramDict={
    'snrThresh':1.,
    'penaltyscale': 1.,
    'useFilterInv':False,  
    'filterName': '',
    'punishFilterName': '', 
    'sigma_n': 1. ,
    'filterMode': "simple",
    'doCLAHE':  False,   
    'inverseSNR': False,
    'demeanMF': False,
    'useGPU':False
        }  
  if typeDict=='silica':
    paramDict['demeanMF'] = True
    paramDict['useFilterInv'] = True
    paramDict['penaltyscale'] = 1.2
    paramDict['doCLAHE'] = True
  elif 'TT' in typeDict:
    paramDict['filterMode'] = 'punishmentFilter'
    paramDict['filterName'] = './myoimages/newSimpleWTFilter.png'
    paramDict['punishFilterName'] = './myoimages/newSimpleWTPunishmentFilter.png'
    # optimized as of June 5, 2018
    paramDict['gamma'] = 3.
    paramDict['snrThresh'] = 0.35
    if '3D' in typeDict:
      # optimized as of December 4, 2018
      paramDict['filterName'] = './myoimages/TT_3D.tif'
      paramDict['punishFilterName'] = './myoimages/TT_Punishment_3D.tif'
      paramDict['snrThresh'] = 0.8
  elif 'LT' in typeDict:
    paramDict['filterMode'] = 'regionalDeviation'
    paramDict['filterName'] = './myoimages/LongitudinalFilter.png'
    # optimized as of June 5, 2018 
    paramDict['snrThresh'] = 0.6 
    paramDict['stdDevThresh'] = 0.2
    if '3D' in typeDict:
      # optimized as of December 4, 2018
      paramDict['filterName'] = './myoimages/LT_3D.tif'
      paramDict['snrThresh'] = 0.9
      paramDict['stdDevThresh'] = 1.
  elif 'TA' in typeDict:
    # optimized as of June 5, 2018
    paramDict['filterMode'] = 'regionalDeviation'
    paramDict['filterName'] = './myoimages/LossFilter.png'
    paramDict['inverseSNR'] = True
    paramDict['snrThresh'] = 0.04 
    paramDict['stdDevThresh'] = 0.1
    if '3D' in typeDict:
      # optimized as of December 4, 2018
      paramDict['filterName'] = './myoimages/TA_3D.tif'
      paramDict['stdDevThresh'] = 0.5

  return paramDict

##
## This function essentially measures overlap between detected regions and hand-annotated regions
## Rules:
## - Hits that do    align with 'truthMarked' are scored   favorably (True positives)
## - Hits that don't align with 'truthMarked' are scored unfavorably (False positives) 
##
def ScoreOverlap_SingleFilter(
          hits,
          truthMarked,                
          mode="default", # negative hits are assessed by 'negativeHits' within positive Hits region
                          # negative hits are penalized throughout entire image 
          display=True):
    # debug
    #truthMarked = np.zeros_like(truthMarked); truthMarked[0:20,:]=1.
    #display=True
    # positive hits 
    masked = np.array(hits > 0, dtype=np.float)
    positiveScoreOverlapImg = truthMarked*masked
    falseMarked = (1-truthMarked) # complement of truth region
    negativeScoreOverlapImg = falseMarked*masked  

    positiveScoreOverlap = np.sum(positiveScoreOverlapImg)/np.sum(truthMarked)
    negativeScoreOverlap = np.sum(negativeScoreOverlapImg)/np.sum(falseMarked)
    
    if display:
      plt.figure()    
      plt.subplot(2,2,1)
      plt.title("All hits")
      plt.imshow(masked)
      plt.subplot(2,2,2)
      plt.title("Truth only")
      plt.imshow(truthMarked)
      plt.subplot(2,2,3)
      plt.title("True positive")
      plt.imshow(positiveScoreOverlapImg)      
      plt.subplot(2,2,4)
      plt.title("False positive")
      plt.imshow(negativeScoreOverlapImg)            
      plt.gcf().savefig("Test.png")
      plt.close()
    
    return positiveScoreOverlap, negativeScoreOverlap

##
## This function essentially measures overlap between detected regions and hand-annotated regions
## Rules:
## - Positive hits that align with 'truthMarked' are scored   favorably (True positives)
## - Negative hits that align with 'truthMarked' are scored unfavorably (False positives) 
##
def ScoreOverlap_CompetingFilters(
          positiveHits,
          negativeHits,
          #positiveTest,               
          truthMarked,                
          mode="default", # negative hits are assessed by 'negativeHits' within positive Hits region
                          # negative hits are penalized throughout entire image 
          display=True):
    # read in 'truth image' 
    #truthMarked = cv2.imread(positiveTest)
    #truthMarked=cv2.cvtColor(truthMarked, cv2.COLOR_BGR2GRAY)
    #truthMarked= np.array(truthMarked> 0, dtype=np.float)
    #imshow(fusedMarked)

    # positive hits 
    positiveMasked = np.array(positiveHits > 0, dtype=np.float)
    if display:
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(positiveMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + positiveMasked
      plt.imshow(composite)
      plt.close()
    #plt.imsave("Test.png",composite)
    positiveScoreOverlapImg = truthMarked*positiveMasked
    positiveScoreOverlap = np.sum(positiveScoreOverlapImg)/np.sum(truthMarked)

    # negative hits 
    negativeMasked = np.array(negativeHits > 0, dtype=np.float)
    if display: 
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(negativeMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + negativeMasked
      plt.imshow(composite)
      plt.close()
    negativeScoreOverlapImg = truthMarked*negativeMasked

    if mode=="default": 
      negativeScoreOverlap = np.sum(negativeScoreOverlapImg)/np.sum(truthMarked)
    elif mode=="nohits":
      dims = np.shape(negativeScoreOverlapImg)
      negativeScoreOverlap = np.sum(negativeScoreOverlapImg)/np.float(np.prod(dims))

    return positiveScoreOverlap, negativeScoreOverlap

##
## Tests a single filter 
##     
def TestParams_Single(
    dataSet,
    paramDict,
    iters = [0,10,20,30,40,50,60,70,80,90],
    display=False):
    # test filter across all angles 
    filter1_filter1Test, dummy = bD.TestFilters(
      testData = dataSet.filter1TestData, 
      filter1Data = dataSet.filter1Data,
      filter2Data = dataSet.filter2Data,
      #subsection=dataSet.filter1TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      #iters = [optimalAngleFused],
      iters=iters,
      colorHitsOutName="filter1Marked_%f.png"%(dataSet.filter1Thresh),
      display=display,
      single=True,
      
      paramDict = paramDict    
    )        
    #dataSet.pasteFilters = False
    if dataSet.pasteFilters:
      hits = painter.doLabel(filter1_filter1Test, dx=dataSet.filter1x,dy=dataSet.filter1y,thresh=0)
    else:
      hits = filter1_filter1Test.stackedHits
    filter1_filter1Test.stackedHits = hits

    # assess score for ROC  
    filter1PS, filter1NS= ScoreOverlap_SingleFilter(
        filter1_filter1Test.stackedHits,
        dataSet.filter1PositiveData,
        display=display)    
        
    print (dataSet.filter1Thresh,filter1PS,filter1NS)
    return filter1PS, filter1NS    
## 
##  Returns true positive/false positive rates for two filters, given respective true positive 
##  annotated data 
## 
def TestParams_Simultaneous(
    dataSet,
    paramDict,
    iters = [0,10,20,30,40,50,60,70,80,90],
    display=False):
   
    ### Filter1 (was fusedPore) 
    #dataSet.iters = [30], # focus on best angle for fused pore data
    ## Test both filters on filter1 test data 
    optimalAngleFused = 30
    print (np.max(dataSet.filter1TestData))
    filter1_filter1Test, filter2_filter1Test = bD.TestFilters(
      testData = dataSet.filter1TestData, 
      filter1Data = dataSet.filter1Data,
      filter2Data = dataSet.filter2Data,
      #subsection=dataSet.filter1TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      #iters = [optimalAngleFused],
      iters=iters,
      colorHitsOutName="filter1Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=display,

      paramDict = paramDict
    )        

    ### Filter2 (was bulkPore) 
    #daImg = cv2.imread(dataSet.name)
    #cut = daImg[dataSet.subsection[0]:dataSet.subsection[1],dataSet.subsection[2]:dataSet.subsection[3]]
    #imshow(cut)

    ## Test both filters on filter2 test data 
    optimalAngleBulk = 5.
    filter1_filter2Test, filter2_filter2Test = bD.TestFilters(
      testData = dataSet.filter2TestData, 
      filter1Data = dataSet.filter1Data,
      filter2Data = dataSet.filter2Data,
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      #iters = [optimalAngleFused],
      colorHitsOutName="filter2Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=display,

      paramDict = paramDict
     )        
    
    # This approach assess the number of hits of filter A overlapping with regions marked as 'A' in the test data
    # negatives refer to hits of filter B on marked 'A' regions
    #if 0:   
    #  fusedPS, bulkNS= ScoreOverlap(filter1_filter1Test.stackedHits,filter2_filter1Test.stackedHits,
    #                       root+"fusedMarked.png", 
    #                       mode="nohits",
    #                       display=display)
#
#      bulkPS, fusedNS = ScoreOverlap(filter2_filter2Test.stackedHits,filter1_filter2Test.stackedHits,
#                            root+"bulkMarked.png",
#                            mode="nohits",
#                            display=display)   
    # This approach assess filter A hits in marked regions of A, penalizes filter A hits in marked regions 
    # of test set B  
    if 1: 
      filter1PS, filter1NS= ScoreOverlap_CompetingFilters(filter1_filter1Test.stackedHits,
			   filter1_filter2Test.stackedHits,
                           dataSet.filter1PositiveData,
                           #negativeTest="testimages/bulkMarked.png", 
                           mode="nohits",
                           display=display)

      filter2PS, filter2NS = ScoreOverlap_CompetingFilters(filter2_filter2Test.stackedHits,			
			    filter2_filter1Test.stackedHits,
                            dataSet.filter2PositiveData,
                            #negativeTest="testimages/fusedMarked.png",
                            mode="nohits",
                            display=display)   
    
    ## 
    print (dataSet.filter1Thresh,dataSet.filter2Thresh,filter1PS,filter2NS,filter2PS,filter1NS)
    return filter1PS,filter2NS,filter2PS,filter1NS

##
## Plots data (as read from pandas dataframe) 
##
def AnalyzePerformanceData(dfOrig,tag='filter1',label=None,normalize=False,roc=True,scale=None,outName=None):
    df = dfOrig
    if scale!=None:
      df=dfOrig[dfOrig.scale==scale]
    

    #plt.figure()
    threshID=tag+'Thresh'
    result = df.sort_values(by=[threshID], ascending=[1])

    if roc:
      f,(ax1,ax2) = plt.subplots(1,2)     
    else: 
      f,(ax1) = plt.subplots(1,1)     
    if label==None:
      title = threshID+" threshold"
    else: 
      title = label+" threshold"

    if scale!=None:
      title+=" scale %3.1f"%scale 
    ax1.set_title(title)  
    if normalize:
      maxNS = np.max( df[tag+'NS'].values ) 
      dfNS=df[tag+'NS']/maxNS
      maxPS = np.max( df[tag+'PS'].values ) 
      dfPS=df[tag+'PS']/maxPS
    else:
      maxNS = 1; maxPS=1.
      dfNS=df[tag+'NS']
      dfPS=df[tag+'PS']

    ax1.scatter(df[threshID], dfPS,label=label+"/positive",c='b')
    ax1.scatter(df[threshID], dfNS,label=label+"/negative",c='r')
    ax1.set_ylabel("Normalized rate") 
    ax1.set_xlabel("Threshold") 
    ax1.set_ylim([0,1]) 
    ax1.set_xlim(xmin=0)
    ax1.legend(loc=0)
    
    if roc==False:
      return 


    ax=ax2   
    ax.set_title("ROC")
    ax.scatter(dfNS,dfPS)
    ax.set_ylim([0,1])
    ax.set_xlim(xmin=0)
    
    # give 50 pct line
    vs = np.linspace(0,1,10)
    ax.plot(vs,vs,'k--')

    i =  np.int(0.45*np.shape(result)[0])
    numbers = np.arange( np.shape(result)[0])
    numbers = numbers[::50]
    #numbers = [i]
   
    for i in numbers:
        #print i
        thresh= result[threshID].values[i]
        ax.scatter(result[tag+'NS'].values[i]/maxNS,result[tag+'PS'].values[i]/maxPS,c="r")
        loc = (result[tag+'NS'].values[i]/maxNS,-0.1+result[tag+'PS'].values[i]/maxPS)
        ax.annotate("%4.2f"%thresh, loc)
    ax.set_ylabel("True positive rate (Normalized)") 
    ax.set_xlabel("False positive rate (Normalized)") 
    plt.tight_layout()
    if outName:
      plt.gcf().savefig(outName,dpi=300)


##
## Iterates over parameter compbinations for TWO filters (simultaneously) to find optimal 
## ROC data 
## 
## dataSet - a dataset object specific to case you're optimizing (see definition) 
## 
## 
def Assess_Simultaneous(
  dataSet,
  paramDict,
  filter1Threshes = np.linspace(800,1100,10), 
  filter2Threshes = np.linspace(800,1100,10), 
  hdf5Name = "optimizer.h5",
  display=False
  ):
  
  # create blank dataframe
  df = pd.DataFrame(columns = ['filter1Thresh','filter2Thresh','filter1PS','filter2NS','filter2PS','filter1NS'])
  
  # iterate of thresholds
  for i,filter1Thresh in enumerate(filter1Threshes):
    for j,filter2Thresh in enumerate(filter2Threshes):
      #for k,penaltyscale      in enumerate(penaltyscales):       
        # set params 
        dataSet.filter1Thresh=filter1Thresh
        dataSet.filter2Thresh=filter2Thresh
#        dataSet.sigma_n = paramDict['sigma_n']
#        dataSet.penaltyscale = paramDict['penaltyscale'] 
#        dataSet.useFilterInv = paramDict['useFilterInv']

        # run test 
        filter1PS,filter2NS,filter2PS,filter1NS = TestParams_Simultaneous(
          dataSet,
          paramDict,
          display=display)

        # store outputs 
        raw_data =  {\
         'filter1Thresh': dataSet.filter1Thresh,
         'filter2Thresh': dataSet.filter2Thresh,
         #'penaltyscale': dataSet.penaltyscale,                
         'filter1PS': filter1PS,
         'filter2NS': filter2NS,
         'filter2PS': filter2PS,
         'filter1NS': filter1NS}
        #print raw_data
        dfi = pd.DataFrame(raw_data,index=[0])#columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
        df=df.append(dfi)

  # store in hdf5 file
  print ("Printing " , hdf5Name )
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     

##
## Test thresholds for a single filter 
## 
def Assess_Single(
  dataSet,
  paramDict,
  filter1Threshes = np.linspace(800,1100,10), 
  hdf5Name = "optimizer_single.h5",
  iters=None,
  display=False
  ):
  
  # create blank dataframe
  df = pd.DataFrame(columns = 
  ['filter1Thresh','filter1PS','filter1NS'])
  
  # iterate of thresholds
  print ("Threshold, Positive Score, Negative Score")
  for i,filter1Thresh in enumerate(filter1Threshes):
        # set params 
        dataSet.filter1Thresh=filter1Thresh
        # run test 
        if iters != None:
          filter1PS,filter1NS = TestParams_Single(
            dataSet,
            paramDict,
            display=display,
            iters=iters)
        else:
          filter1PS,filter1NS = TestParams_Single(
            dataSet,
            paramDict,
            display=display)

        # store outputs 
        raw_data =  {\
         'filter1Thresh': dataSet.filter1Thresh,
         'filter1PS': filter1PS,
         'filter1NS': filter1NS}
        #print raw_data
        dfi = pd.DataFrame(raw_data,index=[0])#columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
        df=df.append(dfi)

  # store in hdf5 file
  print ("Printing " , hdf5Name )
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     


def GenFigROC_TruePos_FalsePos(
  dataSet,
  paramDict,
  filter1Label = "fused",
  f1ts = np.linspace(0.05,0.50,10),
  hdf5Name ="single.hdf5",
  loadOnly=False,
  display=False,
  iters=None
  ):
  ##
  ## perform trials using parameter ranges 
  ##

  if loadOnly:
    print ("Reading ", hdf5Name )
  else:
    Assess_Single(
        dataSet,
        paramDict,
        filter1Threshes = f1ts,
        hdf5Name = hdf5Name,
        display=display,
        iters=iters
      )
      
  ##
  ## Now analyze and make ROC plots 
  ## 
  df = pd.read_hdf(hdf5Name,'table') 
  tag = "filter1"      
  AnalyzePerformanceData(df,tag=tag,label=filter1Label,    
    normalize=True, roc=True,outName=filter1Label+ "_ROC.png")
      
##
## Generates ROC data 
##
def GenFigROC_CompetingFilters(
  dataSet,
  paramDict,
  loadOnly=False,
  filter1Label = "fused",
  filter2Label = "bulk",
  f1ts = np.linspace(0.05,0.50,10),
  f2ts = np.linspace(0.05,0.30,10),
  hdf5Name = "optimizeinvscale.h5"
  ):
  ##
  ## perform trials using parameter ranges 
  ##
  if loadOnly:
    print ("Reading ", hdf5Name )
  else:
    Assess_Simultaneous(
        dataSet,
        paramDict,
        filter1Threshes = f1ts,
        filter2Threshes = f2ts,
        hdf5Name = hdf5Name,
        display=False
      )


  ##
  ## Now analyze and make ROC plots 
  ## 
  df = pd.read_hdf(hdf5Name,'table') 
  tag = "filter1"      
  AnalyzePerformanceData(df,tag=tag,label=filter1Label,    
    normalize=True, roc=True,outName=filter1Label+ "_ROC.png")
  tag = "filter2"          
  AnalyzePerformanceData(df,tag=tag,   label=filter2Label, 
    normalize=True,roc=True,outName=filter2Label+"_ROC.png")



  
  


#!/usr/bin/env python
##################################
#
# Revisions
#       10.08.10 inception
#
##################################


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -optimize" % (scriptName)
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
    if(arg=="-optimize"):
      dataSet = DataSet()
      SetupTests(dataSet) 
      paramDict=ParamDict()
      GenFigROC_CompetingFilters(
        dataSet,
        paramDict,
        f1ts = np.linspace(0.05,0.50,10),   
        f2ts = np.linspace(0.05,0.30,10),   
      ) 
      quit()
    if(arg=="-optimizeLight"):
      dataSet = DataSet()
      SetupTests(dataSet) 
      paramDict=ParamDict()
      GenFigROC_CompetingFilters(
        dataSet,
        paramDict,
        f1ts = np.linspace(0.05,0.50,3),   
        f2ts = np.linspace(0.05,0.30,3),   
      ) 
      # just checking that all still runs 
      print ("PASS")
      quit()
  





  raise RuntimeError("Arguments not understood")





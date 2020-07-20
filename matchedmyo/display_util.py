from __future__ import print_function
'''
This script contains all of the general plotting utilities for this manuscript.
'''

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap


###################################################################################################
###
### Utility Functions Specifically for Figure Generation
###
###################################################################################################

def shaveFig(fileName,padY=None,padX=None,whiteSpace=None):
  '''
  Aggravating way of shaving a figure's white space down to an acceptable level
    and adding in enough whitespace to label the figure
  '''

  img = util.ReadImg(fileName,cvtColor=False)
  imgDims = np.shape(img)

  ### get sum along axis
  rowSum = np.sum(img[:,:,0],axis=1).astype(np.float32)
  colSum = np.sum(img[:,:,0],axis=0).astype(np.float32)

  ### get average along axis
  rowAvg = rowSum / float(imgDims[1])
  colAvg = colSum / float(imgDims[0])

  ### find where the first occurrence of non-white space is
  firstNonWhiteRowIdx = np.argmax((rowAvg-255.)**2. != 0)
  firstNonWhiteColIdx = np.argmax((colAvg-255.)**2. != 0)
  invNonWhiteRowIdx = np.argmax((rowAvg[::-1]-255.)**2. != 0)
  invNonWhiteColIdx = np.argmax((colAvg[::-1]-255.)**2. != 0)

  ### add some padding in
  if padX == None:
    padX = 20
  if padY == None:
    padY = 60 
  firstNonWhiteRowIdx -= padY
  firstNonWhiteColIdx -= padX
  invNonWhiteRowIdx -= padY
  invNonWhiteColIdx -= padX

  idxs = [firstNonWhiteRowIdx, invNonWhiteRowIdx, firstNonWhiteColIdx, invNonWhiteColIdx]
  for i,idx in enumerate(idxs):
    if idx <= 0:
      idxs[i] = 1

  if whiteSpace == None:
    extraWhiteSpace = 100
  else:
    extraWhiteSpace = whiteSpace
  
  newImg = np.zeros((imgDims[0]-idxs[0]-idxs[1]+extraWhiteSpace,
                     imgDims[1]-idxs[2]-idxs[3],3),
                    dtype=np.uint8)

  for channel in range(3):
    newImg[extraWhiteSpace:,:,channel] = img[idxs[0]:-idxs[1],
                                             idxs[2]:-idxs[3],channel]
    newImg[:extraWhiteSpace,:,channel] = 255
  
  cv2.imwrite(fileName,newImg)

def giveAvgStdofDicts(ShamDict,HFDict,MI_DDict,MI_MDict,MI_PDict):
  ### make a big dictionary to iterate through
  results = {'Sham':ShamDict,'HF':HFDict,'MI_D':MI_DDict,'MI_M':MI_MDict,'MI_P':MI_PDict}
  ### make dictionaries to store results
  avgDict = {}; stdDict = {}
  for model,dictionary in results.iteritems():
    ### make holders to store results
    angleAvgs = []; angleStds = []
    for name,angleCounts in dictionary.iteritems():
      print (name)
      if 'angle' not in name:
        continue
      angleAvgs.append(np.mean(angleCounts))
      angleStds.append(np.std(angleCounts))
      print ("Average striation angle:",angleAvgs[-1])
      print ("Standard deviation of striation angle:",angleStds[-1])
    avgDict[model] = angleAvgs
    stdDict[model] = angleStds

  ### Normalize Standard Deviations to Sham Standard Deviation
  ShamAvgStd = np.mean(stdDict['Sham'])
  stdStdDev = {}
  for name,standDev in stdDict.iteritems():
    standDev = np.asarray(standDev,dtype=float) / ShamAvgStd
    stdDict[name] = np.mean(standDev)
    stdStdDev[name] = np.std(standDev)

  ### Make bar chart for angles 
  # need to have results in ordered arrays...
  names = ['Sham', 'HF', 'MI_D', 'MI_M', 'MI_P']
  avgs = []; stds = []
  for name in names:
    avgs.append(avgDict[name])
    stds.append(stdDict[name])
  width = 0.25
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  for i,name in enumerate(names):
    ax.bar(indices+i*width, stdDict[name], width, yerr=stdStdDev[name],ecolor='k',alpha=0.5)
  ax.set_ylabel("Average Angle Standard Deviation Normalized to Sham")
  xtickLocations = np.arange(len(names)) * width + width*3./2.
  ax.set_xticks(xtickLocations)
  ax.set_xticklabels(names,rotation='vertical')
  plt.gcf().savefig("Whole_Dataset_Angles.pdf",dpi=300)

def giveMIBarChart(MI_D, MI_M, MI_P):
  '''
  Gives combined bar chart for all three proximities to the infarct.
  MI_D, MI_M, and MI_P are all dictionaries with structure:
    dict['file name'] = [wtContent, ltContent, lossContent]
  where the contents are floats
  '''

  wtAvgs = {}; wtStds = {}; ltAvgs = {}; ltStds = {}; lossAvgs = {}; lossStds = {};

  DwtC = []; DltC = []; DlossC = []
  for name, content in MI_D.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    DwtC.append(wtC)
    DltC.append(ltC)
    DlossC.append(lossC)
  wtAvgs['D'] = np.mean(DwtC)
  wtStds['D'] = np.std(DwtC)
  ltAvgs['D'] = np.mean(DltC)
  ltStds['D'] = np.std(DltC)
  lossAvgs['D'] = np.mean(DlossC)
  lossStds['D'] = np.std(DlossC)

  MwtC = []; MltC = []; MlossC = [];
  for name, content in MI_M.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    MwtC.append(wtC)
    MltC.append(ltC)
    MlossC.append(lossC)
  wtAvgs['M'] = np.mean(MwtC)
  wtStds['M'] = np.std(MwtC)
  ltAvgs['M'] = np.mean(MltC)
  ltStds['M'] = np.std(MltC)
  lossAvgs['M'] = np.mean(MlossC)
  lossStds['M'] = np.std(MlossC)


  PwtC = []; PltC = []; PlossC = [];
  for name, content in MI_P.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    PwtC.append(wtC)
    PltC.append(ltC)
    PlossC.append(lossC)
  wtAvgs['P'] = np.mean(PwtC)
  wtStds['P'] = np.std(PwtC)
  ltAvgs['P'] = np.mean(PltC)
  ltStds['P'] = np.std(PltC)
  lossAvgs['P'] = np.mean(PlossC)
  lossStds['P'] = np.std(PlossC)

  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 1.0
  N = 11
  indices = np.arange(N)*width + width/4.
  fig,ax = plt.subplots()

  ### plot WT
  rects1 = ax.bar(indices[0], wtAvgs['D'], width, color=colors[0],yerr=wtStds['D'],ecolor='k',label='WT')
  rects2 = ax.bar(indices[1], wtAvgs['M'], width, color=colors[0],yerr=wtStds['M'],ecolor='k',label='WT')
  rects3 = ax.bar(indices[2], wtAvgs['P'], width, color=colors[0],yerr=wtStds['P'],ecolor='k',label='WT')

  ### plot LT
  rects4 = ax.bar(indices[4], ltAvgs['D'], width, color=colors[1],yerr=ltStds['D'],ecolor='k',label='LT')
  rects5 = ax.bar(indices[5], ltAvgs['M'], width, color=colors[1],yerr=ltStds['M'],ecolor='k',label='LT')
  rects6 = ax.bar(indices[6], ltAvgs['P'], width, color=colors[1],yerr=ltStds['P'],ecolor='k',label='LT')

  ### plot Loss
  rects7 = ax.bar(indices[8], lossAvgs['D'], width, color=colors[2],yerr=lossStds['D'],ecolor='k',label='Loss')
  rects8 = ax.bar(indices[9], lossAvgs['M'], width, color=colors[2],yerr=lossStds['M'],ecolor='k',label='Loss')
  rects9 = ax.bar(indices[10],lossAvgs['P'], width, color=colors[2],yerr=lossStds['P'],ecolor='k',label='Loss')

  ax.set_ylabel('Normalized Content')
  ax.legend(handles=[rects1,rects4,rects7])
  newInd = indices + width/2.
  ax.set_xticks(newInd)
  ax.set_xticklabels(['D', 'M','P','','D','M','P','','D','M','P'])
  ax.set_ylim([0,1])
  plt.gcf().savefig('MI_BarChart.pdf',dpi=300)

def giveBarChartfromDict(dictionary,tag):
  ### instantiate lists to contain contents
  wtC = []; ltC = []; lossC = []
  for name,content in dictionary.iteritems():
    if "angle" in name:
      continue
    wtC.append(content[0])
    ltC.append(content[1])
    lossC.append(content[2])

  wtC = np.asarray(wtC)
  ltC = np.asarray(ltC)
  lossC = np.asarray(lossC)

  wtAvg = np.mean(wtC)
  ltAvg = np.mean(ltC)
  lossAvg = np.mean(lossC)

  wtStd = np.std(wtC)
  ltStd = np.std(ltC)
  lossStd = np.std(lossC)

  ### now make a bar chart from this
  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 0.25
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, wtAvg, width, color=colors[0],yerr=wtStd,ecolor='k')
  rects2 = ax.bar(indices+width, ltAvg, width, color=colors[1],yerr=ltStd,ecolor='k')
  rects3 = ax.bar(indices+2*width, lossAvg, width, color=colors[2],yerr=lossStd,ecolor='k')
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  ax.set_ylim([0,1])
  plt.gcf().savefig(tag+'_BarChart.pdf',dpi=300)

def giveAngleHistogram(angleCounts,iters,tag):
  ### Make a histogram for the angles
  iters = np.asarray(iters,dtype='float')
  binSpace = iters[-1] - iters[-2]
  myBins = iters - binSpace / 2.
  myBins= np.append(myBins,myBins[-1] + binSpace)
  plt.figure()
  n, bins, patches = plt.hist(angleCounts, bins=myBins,
                              normed=True,
                              align='mid',
                              facecolor='green', alpha=0.5)
  plt.xlabel('Rotation Angle')
  plt.ylabel('Probability')
  plt.gcf().savefig(tag+"_angle_histogram.pdf",dpi=300)
  plt.close()



"""
Creates stacked image with gradient of transparent reds
"""
# from https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap

def DefineAlphaCmap(daMap="Reds"):

    # Choose colormap
    if daMap is "Reds":
      cmap = plt.cm.Reds
    elif daMap is "Blues":
      cmap = plt.cm.Blues
    else:
      raise RuntimeError("Need to generalize")
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    return my_cmap

def StackGrayRedAlpha(img1,img2,alpha=1.):
    my_cmap = DefineAlphaCmap(daMap="Reds")
    plt.imshow(img1,cmap="gray")
    plt.imshow(img2,cmap=my_cmap,alpha=alpha)

def StackGrayBlueAlpha(img1,img2,alpha=1.):
    my_cmap = DefineAlphaCmap(daMap="Blues")
    plt.imshow(img1,cmap="gray")
    plt.imshow(img2,cmap=my_cmap,alpha=alpha)

def ExampleImage():
    img1=np.array(np.random.rand(100*100)*2e7,dtype=np.uint8)
    img1 = np.reshape(img1,[100,100])

    img2 = np.outer(np.linspace(0,100,101),np.ones(100))

    StackGrayRedAlpha(img1,img2)

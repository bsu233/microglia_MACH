# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %%
import pickle
from sklearn.cluster import DBSCAN
import matplotlib.pylab as plt
import numpy as np 
from scipy import ndimage
import mach_functions
import cv2
import os
import argparse


# %%
# three files

def readresults(idx):
    """
    Readin the pickle objects of the results
    """

    fileNames = ['SmallImages','Results']

    def loadResults(datafile):
        with open(datafile,'rb') as f:
             data = pickle.load(f)
        return data
            
    results = [loadResults("r"+idx+j) for j in fileNames]
    return results



def mergeResults(imgResults,bImage=False,nColumns=10):
    """
    Merge small images into a big image to show the results
    if bImage is true, return a dict contains FIVE binary images
    representing the cell detections
    """
    N = len(imgResults) # by default, there are 100 small images
    nRows = int(np.ceil(N / nColumns))
    
    if bImage == True:
        (dx,dy) = np.shape(imgResults[0]['ram'])
        bigimage = dict()
        for cellType in mach_functions.truthColors.keys():
            tempimage = np.zeros((dx*nRows,dy*nColumns)).astype(np.uint8)
            n = -1
            for i in range(nRows):
                startX=dx*i
                endX = dx*(i+1)
                for j in range(nColumns):
                    n += 1    
                    startY = dy*j
                    endY = dy*(j+1)
                    tempimage[startX:endX,startY:endY] = imgResults[n][cellType]
            
            bigimage[cellType] = tempimage
                        
    else:
        (dx,dy,dummy) = np.shape(imgResults[0])
        bigimage = np.zeros((dx*nRows,dy*nColumns,3)).astype(np.uint8)
        n = -1
        for i in range(nRows):
            startX=dx*i
            endX = dx*(i+1)
            for j in range(nColumns):
                n += 1    
                startY = dy*j
                endY = dy*(j+1)
                bigimage[startX:endX,startY:endY] = imgResults[n]


    return bigimage


def countCluters(bimage,clusterInfoFile,figName,eps=600,minSample=4):
    """
    Count the number of dense clusters using the DBSCAN algorithm.
    """
    lbl = ndimage.label(bimage)[0]
    cs = ndimage.measurements.center_of_mass(bimage, lbl,list(range(1,np.max(lbl)+1)))
    newcs = np.asarray(cs)
    
    db = DBSCAN(eps=eps, min_samples=minSample, metric = 'euclidean',algorithm ='auto')
    db.fit(newcs)
    
    # save the clutering info
    f = open(clusterInfoFile+"cluster",'w')
    nC = np.max(db.labels_) + 1
    
    f.write("Clustering detected cells using the DBSCAN algorithm\n")
    f.write("eps = %d ; minSample = %d \n " % (eps,minSample))
    f.write("=======\n")
    f.write("Total Number of Cluters: %d \n" % (nC))
    
    for i in range(np.max(db.labels_)+1):
        ncell = db.labels_[db.labels_ == i].size
        f.write("Cluster %d: %d cells\n" % (i,ncell))
    f.close()
        
    
    # save cluter figure
    fig, ax = plt.subplots()
    im = ax.scatter(newcs[:,0], newcs[:,1], c=db.labels_)
    plt.gca().invert_yaxis()
    fig.colorbar(im, ax=ax)
    plt.savefig(figName+"cluster.png")


def displayMultImage(images,nColumns=10,cmap=None,numbering=True):
    """
    Function to display the cropped representative microglia cells
    """
    N = len(images)
    if cmap == None:
        # check the if the image is RGB or grayscale
        if len(np.shape(images[0])) == 2:
                cmap='gray'
        else:
                cmap=None

            
        #now of rows, each row has maxi 25 columns
    #nColumns = 25
    nRows = np.ceil(N / nColumns)
    print (nRows)
    #nColumns = 25
        
    figure = plt.figure(figsize=(nRows,nColumns),dpi=300)
            
    for i,img in enumerate(images):
            plt.subplot(nRows,nColumns,i+1)
            plt.imshow(img,cmap=cmap, aspect = "auto")
            plt.xticks([])
            plt.yticks([])
            if numbering:
                plt.title(str(i+1))
            plt.subplots_adjust(wspace=0.01,hspace=0.01)


def Statistics(cellnums,resultfile):
    """
    Get the statistics of the each cell type
    and draw the pie chart
    """
    cells = {"ram": 0,
             "rod": 0,
            "hyp": 0,
            "dys": 0,
            "amoe": 0}
    for i in cells.keys():
        for j in range(100):
            cells[i]  += cellnums[j][i]
    
    colors = ['green','blue','yellow','cyan','red']	
    Allcells = cells['ram'] + cells['hyp'] + cells['dys'] + cells['amoe'] + cells['rod']
    
    sizes = np.asarray([cells['ram'], cells['hyp'], cells['dys'], cells['amoe'], cells['rod']])/Allcells
    plt.figure(figsize=(3,3))    
    plt.pie(sizes, autopct='%1.1f%%',colors=colors,\
        shadow=True, startangle=90)
    plt.savefig("Pie.png")

    f = open(resultfile,'w')
    f.write('%4s\t%4s\t%4s\t%4s\t%4s\n' % ("ram","rod","hyp","dys","amoe"))
    f.write('%4d\t%4d\t%4d\t%4d\t%4d\n' % (cells["ram"],cells["rod"],\
            cells["hyp"],cells["dys"],cells["amoe"]))
    f.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", help="The idx of the image")
    args = parser.parse_args()
    idx = args.idx
      

    # directory of the results 
    resultsdir = os.getcwd()
    os.chdir(resultsdir +"/"+ idx)
    results = readresults(idx)


    imgResults, cellNums = mach_functions.getWholeImageResults(results[1],results[0])

    # save the big image with annotations
    bigresults = mergeResults(imgResults)
    cv2.imwrite("imageresults.png",cv2.cvtColor(bigresults,cv2.COLOR_RGB2BGR))

    # save a binary image for each type of cell
    binaryImage = mergeResults(results[1],bImage=True)
     
    for i in binaryImage.keys():
        np.savetxt(i+"binary",binaryImage[i])

	# do the clustering analysis on the binary image
        countCluters(binaryImage[i],i,i)
    # 

    Statistics(cellNums,"cellnums")


# %%




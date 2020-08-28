

import numpy as np
import matplotlib.pylab as plt
import cv2
from matplotlib import image
import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology
import mach_functions
import pickle


def runFiltering(image,Params,Filters):
    """
    Read in the parameters in the yamlFile and run the hierarchical filtering 
    """

    #directory to store the results
    detectedCells = dict()

    if Params['PenaltyFilterOption'] == 1:
        print ("using the 'rodp' as penalty filter for rod cell detection")
        penaltyFilter = Filters['PenaltyFilters']['rodp']
        rodSNR = Params['rodSNR'][0]
    else:
        print ("using the 'rrodp' as penalty filter for rod cell detection")
        penaltyFilter = Filters['PenaltyFilters']['rrodp']
        rodSNR = Params['rodSNR'][1]
    
    # detect rod and remove the cells in the test image
    print ("Detecting the rod cells ---")
    detectedCells['rod'],crPlanes = mach_functions.giveRod(image,Filters['NormalFilters']['rod'],penaltyFilter,\
							snrthres=rodSNR,fragRodRefine=True,\
							somaFilter=Filters['NormalFilters']['amoe'],
                                                        somathres=0.33,areaRefine=True,
                                                        )
    newtestImage = mach_functions.removeDetectedCells(image,detectedCells['rod'],bgmthres=0.4)

    # detect ram and hyp
    print ("Detecting the ram/hyp cells ---")
    detectedCells['ram'], detectedCells['hyp'] = mach_functions.giveRamHyp(newtestImage,\
							Filters['PartialFilters']['ramp'],\
							Filters['PartialFilters']['hypp'],\
							Params['rampThres'],Params['hyppThres'] )

    newtestImage = mach_functions.removeDetectedCells(newtestImage,detectedCells['ram'],bgmthres=0.4)
    newtestImage = mach_functions.removeDetectedCells(newtestImage,detectedCells['hyp'],bgmthres=0.4)

    # detect amoe and dys
    print ("Detecting the amoe/dys cells ---")
    detectedCells['amoe'],detectedCells['dys'] = mach_functions.giveAmoeDys(newtestImage,Filters['NormalFilters']['amoe'],\
						Params['amoeThres'],Params['areaThres'])


    return detectedCells


def helpmsg():
	scritName = sys.argv[0]
	msg = """
	Purpose: detect microglia cell types in the big image
	Usage: Python scripName -imageYaml -ParYaml
	"""
# %%
# MAIN routine 

if __name__ == "__main__":
	
	image = mach_functions.collectSelecteArea("./unhealthy.yaml")


	yamlFile = "./Params.yaml"
	Params = mach_functions.load_yaml(yamlFile)
	Filters = mach_functions.readInfilters(Params)


	smallImages = mach_functions.divde_big_image(image['grayImage'])
	smallColorImage = mach_functions.divde_big_image(image['rgbImage'])

	allresults = []
	for i,j in enumerate(smallImages):
    		print (f"Small Image ID: {i+1}")
    		tempresults = runFiltering(j,Params,Filters)
    		allresults.append(tempresults)
	
	with open("unhealthyResults","wb") as f:
		pickle.dump(allresults,f)
	
	
	with open("unhealthySmallImages","wb") as f:
		pickle.dump(smallColorImage,f)

	
	with open("unhealthySelectedArea","wb") as f:
		pickle.dump(image['rgbImage'],f)
	






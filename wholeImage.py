
import time
import numpy as np
import matplotlib.pylab as plt
import cv2
import argparse
#from matplotlib import image
import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology
import mach_functions
import pickle


def runFiltering(image,Params,Filters,verbose=False):
    """
    Read in the parameters in the yamlFile and run the hierarchical filtering 
    """

    #directory to store the results
    detectedCells = dict()

    if Params['PenaltyFilterOption'] == 1:
        print ("using the 'rodp' as penalty filter for rod cell detection")
        rodFilter = Filters['NormalFilters']['rod']
        penaltyFilter = Filters['PenaltyFilters']['rodp']
        rodSNR = Params['rodSNR'][0]
    elif Params['PenaltyFilterOption'] == 2:
        print ("using the 'rrodp' as penalty filter for rod cell detection")
        rodFilter = Filters['NormalFilters']['rod']
        penaltyFilter = Filters['PenaltyFilters']['rrodp']
        rodSNR = Params['rodSNR'][1]
    else:
        print ("using the newrod filter (120x120 size)")
        rodFilter = Filters['NormalFilters']['newrod']
        penaltyFilter = Filters['PenaltyFilters']['newrodp']
        rodSNR = Params['rodSNR'][2]
	
	
    
    print ("SNR for rod cell dectection is %f" % (rodSNR) )
    
    # detect rod and remove the cells in the test image
    if verbose:
    	print ("Detecting the rod cells ---")
    detectedCells['rod'],crPlanes = mach_functions.giveRod(image,rodFilter,penaltyFilter,\
							snrthres=rodSNR,fragRodRefine=True,\
							somaFilter=Filters['NormalFilters']['amoe'],
                                                        somathres=0.33,areaRefine=True,
                                                        )
    newtestImage = mach_functions.removeDetectedCells(image,detectedCells['rod'],bgmthres=0.4)

    # detect ram and hyp
    if verbose:
    	print ("Detecting the ram/hyp cells ---")
    detectedCells['ram'], detectedCells['hyp'] = mach_functions.giveRamHyp(newtestImage,\
							Filters['PartialFilters']['ramp'],\
							Filters['PartialFilters']['hypp'],\
							Params['rampThres'],Params['hyppThres'] )

    newtestImage = mach_functions.removeDetectedCells(newtestImage,detectedCells['ram'],bgmthres=0.4)
    newtestImage = mach_functions.removeDetectedCells(newtestImage,detectedCells['hyp'],bgmthres=0.4)

    # detect amoe and dys
    if verbose:
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
	parser = argparse.ArgumentParser()
	parser.add_argument("ImageYaml", help="The Yaml file for the iamge that will be classified")
	parser.add_argument("Prefix", help="The prefix of the result files")
	args = parser.parse_args()
	prefix = args.Prefix

        # read in the image info
	image = mach_functions.collectSelecteArea(args.ImageYaml)
	
	
	start_time = time.time()
	print ("begin filtering ===")

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
	

	print("takes %f hours ---" % ((time.time() - start_time)/3600))

	with open(prefix+"Results","wb") as f:
		pickle.dump(allresults,f)
	
	
	with open(prefix+"SmallImages","wb") as f:
		pickle.dump(smallColorImage,f)

	
	with open(prefix+"SelectedArea","wb") as f:
		pickle.dump(image['rgbImage'],f)
	






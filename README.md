# Using the MACH filter to detect microglia morphology in humain brain tissue images.
The code is still in development. To run the code on our local cluster (faust.luc.edu), you need: 
1. Have a user account on faust. 
2. Since this repo is written in python and uses heavily standard python libaries such as numpy/scipy/opencv
   - Download [Anaconda](https://www.anaconda.com/products/individual#linux)
   - install

3. We now have 227 big images on Faust ()
3. Download this reop (.zip) on to faust.
   - unzip this zip.
   - view the prepare.sh file, 
1) Assign parameters in the "Params.yaml" file
2) Open the "hierarchical_filtering.ipynb" to run the filtering.


Instructions to add:
- logging into faust/accessing the data set
- regenerating binary images from command-line execution of this code
- basic I/O and 'centroid detection' for cells 

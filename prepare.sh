#!/bin/bash

cwd=`pwd`


# this is script to prepare the whole image analysis

#) Step 1, choose which image you want to do the analysis 
#) currently we have 227 image

# suppose we choose image 10
ID=110


# Step 2, create a sub-directory for this image,
# and copy necessary files into this directory

# Note, if the folder "10" exists, please rename it or delete it

cd $cwd && mkdir $ID && cd $ID

cp $cwd/temprun.sh run.sh

cp $cwd/Params.yaml .
cp $cwd/all_images_yaml/image${ID}.yaml .

sed -i 's/XX/'$ID'/g' run.sh
cp $cwd/all_images_yaml/image${ID}.yaml .
sed -i 's/IMAGE/image'$ID'.yaml/g' run.sh
sed -i 's/PREFIX/r'$ID'/g' run.sh


# Step 3) :  Go to the directory, change Parameters in the Params.yaml and image10.yaml
#          and then type "sbatch run.sh" in the terminal to submit the job


# step 4) : after the job is done, type "python analyze.py 10" in the terminal, this will do the 
# analysis

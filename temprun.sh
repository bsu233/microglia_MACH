#!/bin/bash

#SBATCH --job-name='machXX'
#SBATCH --ntasks=1
#SBATCH --mem=10GB


python ../wholeImage.py IMAGE PREFIX

#!/bin/bash

#SBATCH -J job_name
#SBATCH -p dl
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node p100:1
#SBATCH --time=02:00:00

#Load any modules that your program needs
module unload anaconda/python3.8/2020.07
module load deeplearning/2.8.0

#Run your program
srun ./my_program my_program_arguments
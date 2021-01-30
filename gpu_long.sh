#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu3
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=48G  # memory per core (default is 1GB/core)
#SBATCH --time 2-0:0     # days-hours:minutes
#SBATCH --account=general-gpu  # investors will replace this (e.g. `rc-gpu`)
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=gpu_test
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=username@missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

## Module Commands
module load cudnn/cudnn-7.1.4-cuda-9.0.176

## Activate your Python Virtual Environment (if needed)
source /home/jzqm4/my_virtenv3.6.6/bin/activate

# Science goes here:
## CUDA Aware Workflow

# Commands with srun will run on all cores in the allocation
srun python main.py
srun hostname

echo "### Ending at: $(date) ###"

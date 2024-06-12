# Sound Activity-aware Based Cross-task Collaborative Training for Semi-supervised Sound Event Detection


## Introduction

 Briefly, we introduced a Sound Occurrence and Overlap Detection (SOD) task that captures patterns of sound activity to identify non-overlapping or overlapping sounds. And we propose a cross-task collaborative training framework that leverages the relationship between SED and SOD to improve semi-supervised training.


## Get started


1. To reproduce our experiments, please first ensure you have the DESED dataset. For downloading the dataset, please refer to https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline

2. Ensure you have the correct environment. The script `conda_create_environment.sh` is available to create an environment which runs the following code (recommended to run line by line in case of problems).

3. Change all required paths in `confs/sed.yaml` to your own paths.


## Modol training

Run the command `python train_sed.py`  to train the model. 

## Test the trainied model


Run the command `python train_sed.py --test_from_checkpoint YOUR_CHECKPOINT_PATH` to test the trained model. 


## Contact

Please contact Yadong Guan at guanyadonghit@gmail.com for any query.

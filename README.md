# Sound Activity-aware Based Cross-task Collaborative Training for Semi-supervised Sound Event Detection

![](https://img.shields.io/badge/license-MIT-green)

## Introduction

 Briefly, we introduced a Sound Occurrence and Overlap Detection (SOD) task that captures patterns of sound activity to identify non-overlapping or overlapping sounds. And we propose a cross-task collaborative training framework that leverages the relationship between SED and SOD to improve semi-supervised training.


## Get started


1. To reproduce our experiments, please first ensure you have the DESED dataset. For downloading the dataset, please refer to 
[DCASE Task 4 baseline](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline) Please note that the datasets of Dcase Task 4 in different years are slightly different. The dataset we used is from Dcase 2021.

2. Ensure you have the correct environment. The script `conda_create_environment.sh` is available to create an environment which runs the following code (recommended to run line by line in case of problems).

3. Change all required paths in `confs/sed.yaml` to your own paths.


## Train model

Run the command `python train_sed.py`  to train the model. 

## Test the trained model

We provide a [trained model](https://drive.google.com/file/d/1YSebKJ6gbGAri3wXPNEUHW2rKRGMg2nY/view?usp=sharing). The model can be tested using the following command: `python train_sed.py --test_from_checkpoint YOUR_CHECKPOINT_PATH`

### Results of trained model uploaded on DESED Validation dataset:

<img src="[https://img-blog.csdnimg.cn/b937aa6a992d47d9b205f519bcbbc111.png](https://github.com/guanyadong/cross_task_training_sed/assets/49951184/67637018-41c3-4176-ab2e-265fe22b67f0)"  width="600" />


![image](https://github.com/guanyadong/cross_task_training_sed/assets/49951184/67637018-41c3-4176-ab2e-265fe22b67f0=600x)
![sed1](https://github.com/guanyadong/cross_task_training_sed/assets/49951184/af779fbf-8c68-4597-8ab1-e1b6e43d02dc)


## Contact

Please contact Yadong Guan at guanyadonghit@gmail.com for any query.

This repository should include all code required to reproduce the results for "Using Matrix Fisher Distribution to Estimate 3DRotation Probabilities with Deep Learning"

GCC and make needed to run Koev code, this code is not needed for training, just remove the occurances of "loadlibrary" (which loads the .o files to python) in loss.py and run main.py

This code uses conda for its environment.
1. Download conda
2. Create the environment in environment.yml
3. Activate environment

To reproduce results run main.py
1. Configure GPU/CPU to use, I used slurm with only one GPU visible, but if multiple GPUs are visible get the correct one to pytorch
2. Download datasets
3. Configure dataset directory, I had /localstorage/datasets on cluster and ~/datasets on local machine. search and replace magic strings to do this
4. run $python main.py RUN_NAME to start training
5. visualize with tensorboard --logdir logs

6. Download models, and run visualize_.*.py and script_.*.py to get figures and data for paper

If there are major errors contact me, my email is in the paper.

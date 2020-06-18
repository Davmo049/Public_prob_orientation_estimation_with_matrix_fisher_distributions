This repository should include all code required to reproduce the results for "Probabilistic rotation estimation using matrix fisher Distributions"

GCC and make needed to compile Koev code (used to evaluate hypergeometric functions, it is in the c_code directory), this code is not needed for training, just remove the occurances of "loadlibrary" (which loads the .o files to python) in loss.py and run main.py

If one wants to use Koevs code there is a make file which can be used to build it.
Koevs code was slightly modified to run in python instead of matlab.

This code uses conda for its environment.
1. Download conda
2. Create the environment in environment.yml
3. Activate environment

To reproduce results run main.py
1. Configure GPU/CPU to use in main.py, I only had one GPU visible, but if multiple GPUs are visible get the correct one to pytorch
2. Download datasets

Pascal3D: https://cvgl.stanford.edu/projects/pascal3d.html (release 1.1)
Modelnet10: https://github.com/leoshine/Spherical_Regression
UPNA: http://www.unavarra.es/gi4e/databases/hpdb?languageId=1

3. Unzip datasets in dataset directory

4. Configure dataset directory, At the moment it is configured to be in the subdirectory "datasets" in this folder. search and replace magic strings to choose a different directory

4. run $python main.py RUN_NAME
to start training, RUN_NAME is an arbitrary name for your run.

5. visualize with tensorboard --logdir logs

6. Download models, and run visualize_.*.py and script_.*.py to get figures and data for paper, for this you need to reconfigure the directory where the model is loaded from.

There are some files called "test"
Some of them are unittest files, others are files which can be invoked as a main to run and visualize some results.
In addition some files for the dataset subdirectories are used to visualize the preprocessed data with overlaid ground truth.

Some script files saves results to disk or prints them in the terminal, these results were manually copied to other environments to create nicer looking plots (the histograms and the spherical probability visualizations.)

The results you should expect should be

_______________________________________________
Dataset          |  Median error  |   Acc@30   |
_________________|________________|____________|
Pascal3D+        |      8.9       |     90.8   |   (The currently configured training does not use synthetic dataset)
UPNA             |      4.6       |     100.0  |
ModelNet10-SO(3) |     17.1       |     75.7   |   (This median is unstable due to the bimodal nature of the losses for some classes)
_________________|________________|____________|

This code has not been run since I cleaned up dead code and removed identifying strings, removing dead includes might be required.

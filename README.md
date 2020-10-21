![](/github_images/splash_image.png "")

# Introduction

This repository should include all code required to reproduce the results for "Using Matrix Fisher Distribution to Estimate 3DRotation Probabilities with Deep Learning"

# Setting up:
This code uses conda for its environment. To setup this do
1. Download conda
2. Create the environment in environment.yml
3. Activate environment

This code uses datasets. To get them do:
1) create directiory for datasets
2) change string 'dataset' in code in a number of places
  * not the prettiest, but I just needed code to run in one environment
  * to reproduce results you should only need to do this in main.py
3) Download datasets.
  * Pascal3D+:
    * https://cvgl.stanford.edu/projects/pascal3d.html
    * ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
  * Modelnet10-SO(3)
    * https://github.com/leoshine/Spherical_Regression#ModelNet10-SO3-Dataset
    * https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing
  * UPNA
    * http://www.unavarra.es/gi4e/databases/hpdb

# Running code:
To train models run $python main.py
this program takes two arguments
'run_name': a string which will be used for naming your run when logging
'config_file': points to one of the files in the directory configs
for example 'configs/pascal_normal.json'
The training will log with tensorboard format
These logs end up in logs/pascal, logs/modelnet or logs/upna
The training writes the loss in the terminal each epoch as well.

You should get similar performance to what we reported in paper
## Expected Results
![](/github_images/PascalTable.png "")

Expected performance for Pascal3D+

![](/github_images/PascalTable2.png "")

Expected performance for Pascal3D+ top are median error in degrees, bottom in Acc@30 degrees

run unittests by running 
python -m unitttest

To get other data or plots there are a bunch of scripts called "visualize_.*.py" or "script_.*.py"

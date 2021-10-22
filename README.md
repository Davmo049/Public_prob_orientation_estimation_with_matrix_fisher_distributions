![](/github_images/splash_image.png "")

# NOTE
This paper was accepted to NeurIPS 2020 please cite
```
@inproceedings{NEURIPS2020_33cc2b87, 
 author = {Mohlin, David and Sullivan, Josephine and Bianchi, G\'{e}rald},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {4884--4893},
 publisher = {Curran Associates, Inc.},
 title = {Probabilistic Orientation Estimation with Matrix Fisher Distributions},
 url = {https://proceedings.neurips.cc/paper/2020/file/33cc2b872dfe481abef0f61af181dfcf-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
The NeurIPS paper is slightly different than the arxiv paper, mainly how we approximate the hypergeometric normalization constant.

The Pascal3D+ dataset contains samples from imagenet and pascalVOC. In "Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold" they notice that we use the imagenet validation split, while we other works use the Pascal validation split. They run our method with the common datset split and report that our method has average performance of Acc@30: 82.5% and median error of 11.5 degrees. We might rerun experiments and post an update here. In the meantime cite the performance of this method on Pascal3D+ as the numers from "Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold".

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

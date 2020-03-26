# covid-19-cv
X-ray prediction of covid-19 patients


Getting started
---------------
In the project we use Conda for environmenment management and DVC for managing datasets.
To get started:
1) Make sure that conda is installed
2) Install project environment from file
```bash
conda env create -f environment.yml
counda activate covid_19_cv
```
3) Load the datasets from Species Teamdrive.
[DVC](https://dvc.org) is installed by conda, so the only thing that is needed is to pull the datasets with dvc
```
dvc fetch
dvc checkout
```
You must have a team-drive access in order to fetch the file, it may ask to open openauth link to authorize for the first time

Credits
-------

The data is taken from https://github.com/ieee8023/covid-chestxray-dataset
Part of the code is borrowed from https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/
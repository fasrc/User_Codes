Example Jupyter and R notebooks demonstrating machine learning techniques on the classic Iris dataset. These notebooks are used in FASRC training sessions.

### Contents

* [Classification pipeline in Python.ipynb](Classification%20pipeline%20in%20Python.ipynb): End-to-end classification pipeline using scikit-learn and pandas. Covers data exploration, visualization with PCA, feature scaling, feature selection, and training a logistic regression model.

* [Data_Classification.ipynb](Data_Classification.ipynb): K-Nearest Neighbors (KNN) classification example. Demonstrates train/test splitting, feature scaling, model training, and evaluation with confusion matrices.

* [Data_Clustering.ipynb](Data_Clustering.ipynb): K-Means clustering example. Shows unsupervised clustering on the Iris dataset and compares predicted clusters to actual labels.

* [Data_Exploration.Rmd](Data_Exploration.Rmd): R Markdown notebook for data exploration. Demonstrates loading data, train/test splitting, summary statistics, and basic visualizations in R.

* [iris.data](iris.data): The Iris dataset used by the notebooks.



### Getting Started

```bash
git clone https://github.com/fasrc/User_Codes.git
```

## Symlink

```bash
ln -s ScratchLFS_Path Symlink_Name
```

## Conda Env

```bash
conda info --envs
module load Anaconda3
conda create -y -n jupyter_env python=3.6 scikit-learn matplotlib pandas numpy
source activate jupyter_env
```

## R Lib

```bash
getwd()
setwd()
.libPaths()
.libPaths(USER_PATH)
.libPaths( c( USER_PATH , .libPaths() ) )
```

```bash
install.packages("caret", dependencies = c("Depends", "Suggests"))
```

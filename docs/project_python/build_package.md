# Converting available Python scripts into a Python package
Written by: Naeem Khoshnevis   
Research Software Engineer at FASRC   
nkhoshnevis@g.harvard.edu   
Last update: May 18, 2021
-------------------------

Researchers are primarily working with Python modules to carry out their computational needs without any intention of releasing or publishing the codes. As a result, they never feel the need to create a package. Although having several modules and using them directly is not necessarily a bad practice, converting them into a python package significantly improves the installing and maintenance process. Most importantly, it separates the main code from your data and any example code. By converting modules into a package and installing it, you can access your modules in any directory throughout your system, providing you are in the correct environment. In this brief note, we explain the process through the following steps. 

## Step 0: A folder with Python modules

Let's assume we want to create a python package called **mypackage**. We create a folder with the same name. This is the root directory of the package. In this folder, we create another folder with the same name. Please note that the folders have the same name; however, in this note, I will refer to the first folder as *mypackage1* and the second folder (the nested one) as *mypackage2*. If you already have written your modules (e.g., plot.py), you can copy them into the mypackage2 folder. However, if you are starting from scratch, you can keep it empty or add a very simple initial module. Your files and folders structure should be similar to the following figure.


<div align=center>
<img width="300" src="project_python/figure/png/project_python-01.png"/></img>
</div>

## Step 1: Add `setup.py` into mypackage1 folder

`setup.py` is a Python file that includes all required information to build, install, and distribute the modules (read more [here](https://docs.python.org/3/distutils/setupscript.html)). The content of the file can be different for each application. However, a simple `setup.py` file for a simple project can be according to the following code.

```python
from setuptools import setup

setup(
    name="mypackage", 
    version="0.0.1",
    author="your name",
    author_email="your email address",
    description="A brief description about the package",
    license="licence",
    python_requires='>=3.7',
)
```
Now the files and folders structure should be similar to the following figure.

<div align=center>
<img width="300" src="project_python/figure/png/project_python-02.png"/></img>
</div>

## Step 2: Add `requirements.txt` into mypackage1 folder

`requirements.txt` is a file to manage dependencies. This file indicates what packages are required to run the project. If you have already set up your environment and your modules work successfully, you can simply run the following code to create the requirements. 

```python
pip freeze > requirements.txt
```


Now the files and folders structure should be similar to the following figure.

<div align=center>
<img width="300" src="project_python/figure/png/project_python-03.png"/></img>
</div>

## Step 3: Build wheels from setuptools

Python [wheels](https://wheel.readthedocs.io/en/stable/user_guide.html#building-wheels) will build any necessary files and folders that are needed to distribute the package. You can run the following code while inside the mypackage1 folder.

```python
python setup.py bdist_wheel 
```

This should add different folders for distributing source and binary files.

<div align=center>
<img width="300" src="project_python/figure/png/project_python-04.png"/></img>
</div>

## Step 3: Install the package

Now you can install the package and import it into any working directory. The following code installs the package. While you are in mypackage1 folder, run:

```python
pip install -e . 
```
`-e` flag ensures that the package is installed in development mode; as a result, you can debug it, if needed.

This short note covers a very small section of python packaging. However, it is enough to convert your modules to a package that you can install. The [Python Packaging Authority (PyPA)](https://www.pypa.io/en/latest/) is the main resource to learn more about packaging, publishing, and installing Python projects using current tools.
# gReLU

Instructions to install [gReLU](https://github.com/Genentech/gReLU) on
FASRC clusters.

1. Request an interactive job on a GPU partition
   ```
   salloc --partition gpu_test --gres=gpu:1 --mem-per-cpu 2G -c 4 --time 01:00:00
   ```

2. Go to a desired location, e.g., ` cd /n/holylabs/LABS/<PIlab>/LAB/<username>`
   *Note: Remember to replace <username>, wherever it occurs, with your username*
   
3. Create a gReLU project folder, e.g.:
   ```
   mkdir gReLU-proj
   cd gReLU-proj
   ```
4. Clone the gReLU [Git repo](https://github.com/Genentech/gReLU):
   `git clone https://github.com/Genentech/gReLU.git`

5. Load the python module: `module load python`

6. Create a vanilla conda environment with Python 3.11 version at a
desired location using the `--prefix` flag. For example:
   ```
   conda create --prefix=/n/holylabs/LABS/<PIlab>/LAB/<username>/gReLU-proj/gReLUenv python=3.11 -y
   ```

7. Activate the conda environment using the full location of where the environment is stored:
   ```
   source activate /n/holylabs/LABS/<PIlab>/LAB/<username>/gReLU-proj/gReLUenv
   ```

8. Go to the git repo and pip install the software inside that conda environment:
   ```
   cd gReLU
   pip install
   ```

9. pip Install Jupyterlab inside the conda environment: `pip install jupyterlab`

10. Go to OOD, start Jupyterlab app, and load the gReLUenv as a kernel to get started.

11. A jupyter notebook to test out the installation is also attached.

# References:
1. [https://github.com/Genentech/gReLU](https://github.com/Genentech/gReLU)
2. [https://genentech.github.io/gReLU/tutorials/1_inference.html](https://genentech.github.io/gReLU/tutorials/1_inference.html)

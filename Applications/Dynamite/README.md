### Instructions for setting up [Dynamite](https://dynamite.readthedocs.io/en/latest/index.html) in the FASRC Cannon cluster

#### Load the required software modules

```bash
module load python/3.8.5-fasrc01
module load gcc/10.2.0-fasrc01   
module load openmpi/4.1.1-fasrc01
```
#### Download dynamite and pull the development branch

```bash
mkdir ~/sw
cd sw/
git clone https://github.com/GregDMeyer/dynamite.git
cd dynamite/
git pull origin dev
```

#### Download and install PETSc

```bash
cd ~/sw
git clone https://gitlab.com/petsc/petsc.git petsc
cd petsc
git checkout tags/v3.15.0
```

**Note:** You will need to modify the script <code>~/sw/dynamite/petsc_config/complex-opt.py</code> to add the configure option <code>--with-64-bit-indices=1</code>. This is required, if PETSc is intended to be used for very large matrices. Then, you need to configure PETSc with the below commands:

```bash
cd ~/sw/petsc
python ../dynamite/petsc_config/complex-opt.py
```

If all goes well, configure will tell you to run a <code>make</code> command. Copy the command and run it. It should look like: 

```bash
make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all
```

#### Download and install SLEPc

```bash
cd ~/sw
export PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt
git clone https://gitlab.com/slepc/slepc.git slepc
cd slepc
git checkout tags/v3.15.0
./configure
```

If it configures correctly, it will output a <code>make</code> command to run. Copy and paste that, and run it. It should look like:

```bash
make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt
```

#### Building Dynamite

Dynamite is best installed in a <code>conda</code> environment. After you create your environment, you will need to activate it, e.g.:

```bash
conda create -n dynamite python=3.8 pip wheel cython
source activate dynamite
```

You will need to make sure that these environmental variables are all set correctly:

```bash
export PETSC_DIR=<petsc_dir> 
export PETSC_ARCH=complex-opt
export SLEPC_DIR=<your_slepc_installation_directory>
```

Next, install the required Python packages from <copy>requirements.txt</copy>. **Please, notice that here we use the instance from the release dynamite version.**

```bash
cd ~/sw/dynamite
# Get the requirements.txt from the release version
wget https://raw.githubusercontent.com/GregDMeyer/dynamite/master/requirements.txt
pip install -r requirements.txt
```

Then build dynamite:

```bash
cd ~/sw/dynamite
pip install -e ./
```




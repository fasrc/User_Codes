# Installing Software Yourself

We have high-performance, scientific computing applications installed and available through our [software module system](https://docs.rc.fas.harvard.edu/kb/modules-intro/). These include compilers, licensed software, and commonly used libraries.

If you are looking to use a "bleeding edge" version of software or something specific only to your lab, you can install applications yourself in your home directory, any other space to which you can write, or a shared lab directory. Writing and installing your own software may be part of your research, too. Below you will find several approaches to compiling and installing personal software on the cluster. If you are having problems during your installation, submit a [help request on the RC Portal](https://portal.rc.fas.harvard.edu/rcrt/submit_ticket) or stop by [Office Hours](https://www.rc.fas.harvard.edu/training/office-hours/).

* [Installing binaries](Binaries.md)
* [Installing GNU-toolchain-style apps (configure - make - make install)](Gnu.md)
* Cmake
* [Spack](Spack.md)

## Making Software Available

Once you have built the software you want to use you need to make it available for use. The shell environment relies on environment variables that name where apps and libraries are located. When you install software to non-default locations, you will need to update these variables. Without doing so, you will get errors like command not found, or error while loading shared libraries. Note that if you are using [Spack](Spack.md), [Python](../../Languages/Python/Mamba.md), [R](../../Languages/R/README.md), or [Julia](../../Languages/Julia/README.md) you will want to follow the guides for those codes as they have their own ways of handling software.

For example, the environment variable <code>PATH</code> names directories in which the apps you invoke on the command line reside. Conventionally, when you install software under a <code>--prefix</code>, the apps are in a sub-directory of that prefix named <code>bin</code>. Similarly, the variable <code>LD_LIBRARY_PATH</code> controls what libraries are available for dynamic dependency loading, and those libraries are often put in a directory named <code>lib</code>. If you installed your software to <code>/n/holylabs/LABS/jharvard_lab/Lab/software</code> the variables you will need will likely be:

```bash
export PATH="/n/holylabs/LABS/jharvard_lab/Lab/software/bin:$PATH"
export LD_LIBRARY_PATH="/n/holylabs/LABS/jharvard_lab/Lab/software/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/n/holylabs/LABS/jharvard_lab/Lab/software/lib:$LIBRARY_PATH"
export PKG_CONFIG_PATH="/n/holylabs/LABS/jharvard_lab/Lab/software/lib/pkgconfig:$PKG_CONFIG_PATH"
export CPATH="/n/holylabs/LABS/jharvard_lab/Lab/software/include:$CPATH"
export FPATH="/n/holylabs/LABS/jharvard_lab/Lab/software/include:$FPATH"
export PYTHONPATH="/n/holylabs/LABS/jharvard_lab/Lab/software/lib/python2.7/site-packages:$PYTHONPATH"
export MANPATH="/n/holylabs/LABS/jharvard_lab/Lab/software/share/man:$MANPATH"
```

Once you find all of these you can copy these variable definitions to the end of your <code>~/.bashrc</code> file, and then the software will be available by default. Be sure to include the <code>:$VARIABLE</code> at the end else you will reset the variable rather than prepending. The operating system will search the list of folders defined in the variable in order to find the package or library it is looking for.  Alternatively, you can save this output to a file such as ~/sw/setup.sh, and then every time you want to use this software, you run:

```bash
source ~/sw/setup.sh
```

This is necessary if you want to install multiple incompatible apps, or different versions of the same app, and pick and choose between them at will (you'll have to use a different <code>--prefix</code> for each).
You may also consider writing your own [lmod modulefiles](https://lmod.readthedocs.io/en/latest/015_writing_modules.html) so that you can [integrate](https://lmod.readthedocs.io/en/latest/020_advanced.html) your own software with our modules system.
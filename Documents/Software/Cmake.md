## Cmake

[Cmake](https://cmake.org/) is a popular software compilation tool that allows the building of binaries and libraries for various operating systems and environments. Cmake works in a similar but notably different way to the [GNU autotools](Gnu.md) build process. Of importance is to make sure that the version of Cmake you are using is up to date as newer versions have newer features which the software you are building may be leveraging.  We provide prebuilt versions of Cmake as part of our module system and you can find what versions are available by doing:

```bash
$ module spider cmake

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  cmake: cmake/3.25.2-fasrc01
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Description:
      Cross platform build tool


    This module can be loaded directly: module load cmake/3.25.2-fasrc01

    Help:
      cmake-3.25.2-fasrc01
      Cross platform build tool

$ module load cmake/3.25.2-fasrc01
```

Once you have loaded the module you need for Cmake check the build/installation instructions for your code as how you use Cmake will depend on where the developer has put their <code>Cmakelists.txt</code> which specifies the build process. We will proceed assuming that they have placed it in the base directory of their package. If this is the case you will want to <code>cd</code> to that directory and <code>mkdir build</code>, this directory will serve as the build directory which Cmake will use for its intermediate build steps.  You then:

```bash
cd build
cmake -DCMAKE_INSTALL_PREFIX=/n/holylabs/LABS/jharvard_lab/Lab/software ..
make
```

This will run Cmake and specify a location you want to install the resulting software to. Cmake when its done configuring then generates a <code>Makefile</code> that you then run via <code>make</code>. Once the code is compiled you can then install the software by doing:

```bash
make install
```

It should be noted that Cmake sometimes does not respect the <code>CMAKE_INSTALL_PREFIX</code> flag. In this case you will need to copy the created libraries by hand, you can find the files that Cmake made in the <code>build</code> directory.
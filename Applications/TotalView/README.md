## TotalView

<img src="figures/totalview-logo.jpeg" alt="TotalView-logo" width="300"/>

### What is TotalView?

[TotalView](https://totalview.io/) from [PERFORCE](https://www.perforce.com/) is a debugging tool particularly suitable for debugging of parallel applications. It provides both X Window-based Graphical User Interface (GUI) and command line interface (CLI) environments for debugging. This example illustrates how to use TotalView on the FAS Cannon cluster.

The specific example, <code>wave_mpi.f90</code>, solves the wave equation in parallel using MPI.

### Contents:

<code>wave_mpi.f90</code>: Example MPI Fortran code
                       
### Using TotalView on Cannon:

To use Totalview on Cannon, first you need to load the Totalview module-file to set the correct environment settings. This is done most conveniently by placing the command <code>module load totalview/2019.0.4-fasrc01</code> in your <code>.bashrc</code> startup file.

In order to debug MPI parallel applications, in addition to the TotalView software module, you need also to load appropriate Compiler and MPI modules, for instance:

```bash
module load totalview/2019.0.4-fasrc01
module load intel/21.2.0-fasrc01
module load openmpi/4.1.3-fasrc01
```

To use Totalview, your code must be compiled with the <code>-g</code> compile option. It is also recommended to suppress any level of optimization by compiling your application with the <code>-O0</code> option.

<pre>
<font style="color:red">Fortran 77:</font> mpif77 -g -O0 -o tv_test.x tv_test.f
<font style="color:red">Fortran 90:</font> mpif90 -g -O0 -o tv_test.x tv_test.f90
<font style="color:red">C:</font>          mpicc -g -O0 -o tv_test.x tv_test.c
<font style="color:red">C++:</font>        mpicxx -g -O0 -o tv_test.x tv_test.cpp
</pre>

**Note:** The instrumented executable should be used for debugging only, not in production runs. After your code is up and running, for actual production runs you need to recompile your application with the desired level of optimization.

To use Totalview, you need to log in with an X window forwarding enabled. If you access Cannon from a Unix-like system, you have to use the -X or -Y option to ssh. The -Y option often works better for Mac OS X. For instructions on how to enable X11 forwarding when accessing Cannon from Windows click [here](https://docs.rc.fas.harvard.edu/kb/x11-forwarding/).

```bash
ssh -l username login.rc.fas.harvard.edu -YC
```

**Note:** Alternatively, one may use the [VDI portal](https://vdi.rc.fas.harvard.edu/pun/sys/dashboard/batch_connect/sessions) and then launch a "Remote Desktop" app.  

After loading the Totalview module and compiling with the -g option, request an interactive session:

```bash
salloc -p test --x11 -n 4 -t 00-02:00 --mem-per-cpu=4000
```
This will start an interactive (bash) shell and load the module-files included in your startup .bashrc file. Then launch the debugger with one of the following commands:

```bash
totalview mpirun -a -np 4 ./tv_test.x
```

or

```bash
mpirun -np 4 -tv ./tv_test.x
```

or

```bash
mpirun -np 4 -debug ./tv_test.x
```

The TotalView startup GUI will pop up and display debugging startup parameters, as illustrated below. After reviewing them, click OK.

![tv1](figures/tv1.png)

Go to the process window, and click the "Go" button.

![tv2](figures/tv2.png)

**Note:** At this stage Totalview displays the source code of the mpirun function, NOT the source code of your application.

After you click "GO" in the process window, a small window will pop up, asking whether the mpirun process should be stopped. Click "Yes".

![tv3](figures/tv3.png)

Then, in the "Stack Trace" section of the process window you should see the name of the main program of your application. You can now display the source code by clicking on it. To start debugging, create a break point by clicking on a line number in the source pane, and click "Go". After that, you can use other buttons ("Next", "Step", "Out", etc).

![tv4](figures/tv4.png)

### References:

* [Official Totalview Documentation](https://help.totalview.io/current/HTML/index.html)
* [Totalview tutorial, Lawrence Livermore National Laboratory](https://hpc.llnl.gov/documentation/tutorials/totalview-tutorial)
* [Totalview tutorial - common functions, LLNL](https://hpc.llnl.gov/documentation/tutorials/totalview-part-2-common-functions)

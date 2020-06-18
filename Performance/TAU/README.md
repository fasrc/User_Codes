## TAU - Tuning and Analysis Utilities

![TAU Logo](Images/tau-logo.png)

[TAU (Tuning and Analysis Utilities)](https://www.cs.uoregon.edu/research/tau/home.php) is a comprehensive profiling and tracing toolkit for performance analysis of parallel programs written in Fortran, C, C++, Java, and Python. It is capable of gathering performance information through instrumentation of functions, methods, basic blocks, and statements. All C++ language features are supported including templates and namespaces. The instrumentation consists of calls to TAU library routines, which can be incorporated into a program in several ways:

* Automatic instrumentation using the compiler
* Automatic instrumentation using the Program Database Toolkit (PDT)
* Manual instrumentation using the instrumentation API
* At runtime using library call interception through the <code>tau_exec</code> command
* Dynamically using DyninstAPI
* At runtime in the Java virtual machine

Data Analysis and Visualization:

* Profile data: TAU's profile visualization tool, ParaProf, provides a variety of graphical displays for profile data to help users quickly identify sources of performance bottlenecks. The text based <code>pprof</code> tool is also available for analyzing profile data.
* Trace data: TAU provides the JumpShot trace visualization tool for graphical viewing of trace data. TAU also provide utilities to convert trace data into formats for viewing with Vampir, Paraver and other performance analysis tools.

**Programming models and platforms:** TAU supports most commonly used parallel hardware and programming models, including Intel, Cray, IBM, Sun, Apple, SGI, GPUs/Accelerators, HP, NEC, Fujitsu, MS Windows, using MPI, OpenMP, Pthreads, OpenCL, CUDA and Hybrid.

To get started with TAU on the FAS cluster you can try the below examples:

* [Example 1](Example1): Profiling mpi4py

#### References:

* [TAU website](https://www.cs.uoregon.edu/research/tau/home.php)
* [Official Documentation](https://www.cs.uoregon.edu/research/tau/docs.php)
## Julia

![Julia Logo](Images/julia-logo.png)

[Julia](https://en.wikipedia.org/wiki/Julia_(programming_language)) is a high-level, high-performance dynamic programming language for technical computing. It has syntax that is familiar to users of many other technical computing environments. Designed at MIT to tackle large-scale partial-differential equation simulation and distributed linear algebra, Julia features a robust ecosystem of tools for
[optimization,](https://www.juliaopt.org/)
[statistics,](https://juliastats.github.io/)
[parallel programming,](https://julia.mit.edu/#parallel) and 
[data visualization.](https://juliaplots.github.io/)
Julia is actively developed by teams
[at MIT](https://julia.mit.edu/) and 
[in industry,](https://juliacomputing.com/) along with 
[hundreds of domain-expert scientists and programmers from around the world](https://github.com/JuliaLang/julia/graphs/contributors).


![Gadfly Demo](Images/gadfly-demo.png)

### Installation
Julia most easily [installed](https://docs.julialang.org/en/v1/manual/getting-started/) by downloading the precompiled [binaries](https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Binaries.md) from: [https://julialang.org/downloads/](https://julialang.org/downloads/) We recommend the Generic Linux on x86 64 bit glibc version. This avoids any need to build any dependencies. We recommend downloading it to your holylabs directory for use.

Once you have downloaded Julia you can add Julia to your path by setting the following in your <code>~/.bashrc</code>

```bash
export PATH=$PATH:/n/holylabs/LABS/jharvard_lab/Lab/software/julia-1.9.3/bin
```

Subbing in whereever you extracted Julia to. Once that is set the next time you login Julia will be in your path and ready to use.

#### Example Installation

This is an example for Julia 1.9.3. Check [Julia downloads](https://julialang.org/downloads/) for other versions.

```bash
# use lab storage
[jharvard@holy7c12104 ~]$ cd /n/holylabs/LABS/jharvard_lab/Users/jharvard/software/

# download julia and extract
[jharvard@holy7c12104 software]$ wget  https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
[jharvard@holy7c12104 software]$ tar xvfz julia-1.9.3-linux-x86_64.tar.gz

# add julia to path
[jharvard@holy7c12104 julia-1.9.3]$ export PATH=$PATH:/n/holylabs/LABS/jharvard_lab/Users/jharvard/software/julia-1.9.3/bin

[jharvard@holy7c12104 julia-1.9.3]$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.3 (2023-08-24)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```

To get started with Julia on the FAS cluster you can try the below examples:

* [Example 1](Example1): Monte-Carlo calculation of PI
* [Example 2](Example2): Ordinary Differential Equations (ODEs)

You can also use Julia in [Jupyter notebooks in our VDI interactive environment](Notebook.md).

#### References:

* [The Julia Programming Language](https://julialang.org/)
* [Julia Computing](https://juliacomputing.com/)
* [Julia Documentation](https://docs.julialang.org/en/v1/)



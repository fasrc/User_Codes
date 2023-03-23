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

To get started with Julia on the FAS cluster you can try the below examples:

* [Example 1](Example1): Monte-Carlo calculation of PI
* [Example 2](Example2): Ordinary Differential Equations (ODEs)

You can also use Julia in [Jupyter notebooks in our VDI interactive environment](Notebook.md).

![Gadfly Demo](Images/gadfly-demo.png)

### Installation
Julia most easily [installed](https://docs.julialang.org/en/v1/manual/getting-started/) by downloading the precompiled binares from: [https://julialang.org/downloads/](https://julialang.org/downloads/) We recommend the Generic Linux on x86 64 bit glibc version. This avoids any need to build any dependencies. Julia is a fairly big package so we recommend downloading it to your holylabs directory for use.

Once you have downloaded Julia you can add Julia to your path by setting the following in your <code>~/.bashrc</code>

```bash
export PATH=$PATH:/n/holylabs/LABS/jharvard_lab/Lab/software/julia-1.8.5/bin
```

Subbing in whereever you extracted Julia to. Once that is set the next time you login Julia will be in your path and ready to use.



#### References:

* [The Julia Programming Language](https://julialang.org/)
* [Julia Computing](https://juliacomputing.com/)
* [Julia Documentation](https://docs.julialang.org/en/v1/)



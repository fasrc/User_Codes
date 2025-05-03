# MPI with Julia on FASRC

### Installing Julia

First, install Julia on the cluster using [juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh
```

### Creating an environment for MPI

```bash
cd ~
mkdir MPIenv
```

Then in julia, enter Package Manager Mode: Type <code>]</code> to enter the package manager mode. Run

```bash
(@v1.11) pkg> activate ~/MPIenv
(@v1.11) pkg> add MPI MPIPreferences
```

to install [MPI.jl](https://github.com/JuliaParallel/MPI.jl). After creating this environment, any parallel script can be run in this environment, for example:

```bash
mpiexec -n 4 julia --project=~/MPIenv script.jl
```
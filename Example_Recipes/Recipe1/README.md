#### Purpose:

[Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO) with [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) objective function. The specific implementation uses the [pso R package](https://cran.r-project.org/web/packages/pso/). 

**Note:** Example assumes the *pso* R package is already [installed locally in user environment](https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/r/).

#### Contents:

* <code>pso_rosenbrock.R</code>: R source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>pso_rosenbrock.Rout</code>: Output file

#### R source code:

```r
#==========================================================
# Rosenbrock function
#==========================================================
fnRosenbrock <- function (x) {
  n <- length(x)
  x1 <- x[2:n]
  x2 <- x[1:(n - 1)]
  sum(100 * (x1 - x2^2)^2 + (1 - x2)^2)
}

#==========================================================
# PSO
#==========================================================
pso_rosen <- function(){
  library("pso")
  
  # +++ PSO parameters +++
  nvars        <- 2       # dimensions (number of variables)
  nps          <- 4       # number of particles
  niter        <- 5000    # number of iterations
  lower_bounds <- -3      # lower bounds
  upper_bounds <-  3      # upper bounds  
  reltol_value <- 1e-15
  abstol_value <- 1e-12
  # +++ END PSO parameters +++ 

  set.seed(1)
  fn  <- fnRosenbrock
  gr  <- fnRosenbrock
  sol <- psoptim( rep(NA, nvars), 
                  fn, 
                  gr, 
                  lower = lower_bounds,
                  upper = upper_bounds, 
                  control = list( reltol = reltol_value,
                                  abstol = abstol_value,
                                  maxit = niter, 
                                  s = nps, 
                                  trace = 1,
                                  REPORT = 100,
                                  vectorize = FALSE ) )

  print(sol$par)
  print(sol$val)
  print(sol$counts)
  print(sol$message)
}

# +++ optimize +++
pso_rosen()
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pso_test
#SBATCH -o pso_test.out
#SBATCH -e pso_test.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Set up software environment
module load R/3.5.1-fasrc01
export R_LIBS_USER=$HOME/software/R/3.5.1:$R_LIBS_USER

# Run program
srun -n 1 -c 1 R CMD BATCH --no-save --no-restore pso_rosenbrock.R  
```

**Note:** This assumes the *pso* R package is installed at <code>$HOME/software/R/3.5.1</code>


#### Example Usage:

```bash
module load R/3.5.1-fasrc01
sbatch run.sbatch
```
#### Example Output:

```
$ cat pso_rosenbrock.Rout

R version 3.5.1 (2018-07-02) -- "Feather Spray"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> #==========================================================
> # Rosenbrock function
> #==========================================================
> fnRosenbrock <- function (x) {
+   n <- length(x)
+   x1 <- x[2:n]
+   x2 <- x[1:(n - 1)]
+   sum(100 * (x1 - x2^2)^2 + (1 - x2)^2)
+ }
> 
> #==========================================================
> # PSO
> #==========================================================
> pso_rosen <- function(){
+   library("pso")
+   
+   # +++ PSO parameters +++
+   nvars        <- 2       # dimensions (number of variables)
+   nps          <- 4       # number of particles
+   niter        <- 5000    # number of iterations
+   lower_bounds <- -3      # lower bounds
+   upper_bounds <-  3      # upper bounds  
+   reltol_value <- 1e-15
+   abstol_value <- 1e-12
+   # +++ END PSO parameters +++ 
+ 
+   set.seed(1)
+   fn  <- fnRosenbrock
+   gr  <- fnRosenbrock
+   sol <- psoptim( rep(NA, nvars), 
+                   fn, 
+                   gr, 
+                   lower = lower_bounds,
+                   upper = upper_bounds, 
+                   control = list( reltol = reltol_value,
+                                   abstol = abstol_value,
+                                   maxit = niter, 
+                                   s = nps, 
+                                   trace = 1,
+                                   REPORT = 100,
+                                   vectorize = FALSE ) )
+ 
+   print(sol$par)
+   print(sol$val)
+   print(sol$counts)
+   print(sol$message)
+ }
> 
> # +++ optimize +++
> pso_rosen()
S=4, K=3, p=0.5781, w0=0.7213, w1=0.7213, c.p=1.193, c.g=1.193
v.max=NA, d=8.485, vectorize=FALSE, hybrid=off
It 100: fitness=0.09605, swarm diam.=0.02149
It 200: fitness=0.05159, swarm diam.=0.04315
It 300: fitness=0.03826, swarm diam.=0.006891
It 400: fitness=0.0257, swarm diam.=0.01715
It 500: fitness=0.01002, swarm diam.=0.001284
It 600: fitness=0.005642, swarm diam.=0.03376
It 700: fitness=0.003919, swarm diam.=0.0009955
It 800: fitness=0.001952, swarm diam.=0.0001496
It 900: fitness=0.0006658, swarm diam.=0.001239
It 1000: fitness=0.0005059, swarm diam.=0.003053
It 1100: fitness=0.0002355, swarm diam.=0.0001606
It 1200: fitness=0.0001102, swarm diam.=0.0002106
It 1300: fitness=8.255e-05, swarm diam.=0.0002729
It 1400: fitness=3.116e-05, swarm diam.=3.908e-05
It 1500: fitness=1.648e-05, swarm diam.=0.0002789
It 1600: fitness=1.493e-05, swarm diam.=0.0003664
It 1700: fitness=3.384e-06, swarm diam.=0.00118
It 1800: fitness=2.633e-06, swarm diam.=0.0002803
It 1900: fitness=9.617e-07, swarm diam.=0.0002946
It 2000: fitness=6.693e-07, swarm diam.=7.499e-06
It 2100: fitness=3.927e-07, swarm diam.=3.696e-05
It 2200: fitness=2.984e-07, swarm diam.=7.419e-06
It 2300: fitness=1.067e-07, swarm diam.=1.246e-06
It 2400: fitness=7.928e-08, swarm diam.=3.538e-06
It 2500: fitness=7.419e-08, swarm diam.=1.5e-05
It 2600: fitness=2.585e-08, swarm diam.=1.289e-05
It 2700: fitness=1.188e-08, swarm diam.=1.15e-06
It 2800: fitness=5.245e-09, swarm diam.=7.598e-06
It 2900: fitness=1.884e-09, swarm diam.=1.281e-05
It 3000: fitness=1.069e-09, swarm diam.=1.956e-07
It 3100: fitness=1.062e-09, swarm diam.=4.84e-07
It 3200: fitness=1.023e-09, swarm diam.=5.502e-08
It 3300: fitness=6.87e-10, swarm diam.=5.194e-06
It 3400: fitness=4.086e-10, swarm diam.=4.602e-07
It 3500: fitness=1.543e-10, swarm diam.=2.212e-06
It 3600: fitness=1.133e-10, swarm diam.=8.777e-07
It 3700: fitness=9.857e-11, swarm diam.=8.293e-07
It 3800: fitness=4.551e-11, swarm diam.=1.917e-06
It 3900: fitness=4.076e-11, swarm diam.=2.353e-07
It 4000: fitness=2.965e-11, swarm diam.=9.379e-08
It 4100: fitness=2.684e-11, swarm diam.=3.057e-08
It 4200: fitness=1.468e-11, swarm diam.=1.144e-07
It 4300: fitness=8.61e-12, swarm diam.=2.704e-07
It 4400: fitness=5.922e-12, swarm diam.=8.899e-08
It 4500: fitness=4.101e-12, swarm diam.=9.718e-08
It 4600: fitness=1.679e-12, swarm diam.=1.006e-07
Converged
[1] 0.999999 0.999998
[1] 9.965205e-13
 function iteration  restarts 
    18724      4681         0 
[1] "Converged"
> 
> proc.time()
   user  system elapsed 
  0.640   0.075   1.367 
```
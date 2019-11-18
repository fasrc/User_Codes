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
# PSO: This uses the PSO algorithms from *metaheuristicOpt*
#==========================================================
pso_rosen <- function()
{
  # +++ Parameters +++
  seed         <- 1        # random generator seed
  dim          <- 2        # dimensions (number of variables)
  nps          <- 4        # number of particles
  niter        <- 2000     # number of iterations
  lower_bounds <- -3       # lower bounds
  upper_bounds <-  3       # upper bounds
  inertia      <- 0.7298   # weights
  c1           <- 1.49618 
  c2           <- 1.49618

  # +++ Set up problem +++
  fn       <- fnRosenbrock
  numVar   <- dim
  rangeVar <- matrix(c(lower_bounds, upper_bounds), nrow=numVar)
  control  <- list(numPopulation=nps, maxIter=niter, Vmax=2, ci=c1, cg=c2, w=inertia)

  sol <- metaOpt( fn,                    # Objective function
                  optimType = "MIN",     # Optimization type
                  algorithm = "PSO",     # Optimization algorithm
                  numVar,                # Problem dimension
                  rangeVar,              # Varible bounds
                  control,               # Optimization parameters
                  seed )                 # Seed for random number generator
  
  # +++ Print out results +++
  print(sol[1])
  print(sol[2])
  print(sol[3])
}

# +++ Optimize +++
library("metaheuristicOpt")
pso_rosen()

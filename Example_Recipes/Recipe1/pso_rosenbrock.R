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

##       author: Naeem Khoshnevis
##      created: April 2022
##      purpose: Estimating PI

# references
# https://helloacm.com/r-programming-tutorial-how-to-compute-pi-using-monte-carlo-in-r/
# pi : http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
# PI : 3.14159265358979323846

PI <- 3.14159265358979323846

mc_pi <- function(sample_size){
  
  set.seed(as.integer(proc.time()[[3]]*1000))
  x <- runif(sample_size)
  y <- runif(sample_size)
  z <- sqrt(x^2+y^2)
  pi <- (length(which(z<=1))*4)/length(z)

  return(pi)
}

match_chars <- function(number_1, number_2){
  
  sn1 <- strsplit(sprintf("%.54f", number_1),"")[[1]]
  sn2 <- strsplit(sprintf("%.54f", number_2),"")[[1]]
  
  i <- 1
  l <- min(length(sn1), length(sn2))
  
  for (i in seq(1,l)){
    if (sn1[i] != sn2[i]){
      return(i-1)
    } else {
      i <- i + 1
    }
  }
  return(i-1)
}
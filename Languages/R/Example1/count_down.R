#===========================================================
# Program: count_down.R
#
# Run:     R --vanilla < count_down.R         
#===========================================================

# Function CountDown........................................
CountDown <- function(x)
{
  print( x )
  while( x != 0 )
  {
    Sys.sleep(1)
    x <- x - 1
    print( x )
  }
}

# Call CountDown............................................
CountDown( 10 )

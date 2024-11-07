#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: pi_monte_carlo.jl
#          Monte-Carlo calculation of PI
#
# Usage: julia pi_monte_carlo.jl
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function montepi(n::Int)
   R = 1.0
   s = 0
   for i = 1: n
      x = R * rand()
      y = R * rand()
      if x^2 + y^2 <= R^2
         s = s + 1
      end
   end
   return 4.0*s/n
end

# Main program
for i in 3: 8
    n = 10^i
    p = montepi(n)
    println("N = $n: PI = $p")
end

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: ode_test.jl 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
using DifferentialEquations
using SimpleDiffEq
using Printf
using Plots
using Plots.PlotMeasures
using PyCall

# --- Define the problem ---
f(u, p, t) = t - u
u0 = 1.0             # Initial conditions
tspan = (0.0, 5.0)   # Time span
prob = ODEProblem(f, u0, tspan)

# --- Solve the problem ---
dt  = 0.2
alg = RK4()
sol = solve(prob, alg, saveat=dt)

# --- Exact solution ---
function f_ex(t)
   f = t - 1.0 + ( 2.0 * exp(-t) )
   return f
end

# --- Print out solution ---
fname = "results.dat"
fo = open(fname, "w")
println(fo, "   Time     Numeric     Exact   ")
println(fo, " ------------------------------ ")
for i = 1: size(sol.t)[1]
   x = sol.t[i]
   y = sol.u[i]
   y_exact = f_ex(x)
   @printf(fo, "%8.4f %10.6f %10.6f\n", x, y, y_exact)
end
close(fo)

pyimport("numpy")

# --- Run the Python script to plot the result ---
@pyinclude("figure.py")
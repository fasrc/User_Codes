import gurobipy as gp
from gurobipy import GRB

# tested with Python 3.7.0 & Gurobi 9.1.0
# Sample data: values of independent variable x and dependent variable y
observations, x, y = gp.multidict({
    ('1'): [0,1],
    ('2'): [0.5,0.9],
    ('3'): [1,0.7],
    ('4'): [1.5,1.5],
    ('5'): [1.9,2],
    ('6'): [2.5,2.4],
    ('7'): [3,3.2],
    ('8'): [3.5,2],
    ('9'): [4,2.7],
    ('10'): [4.5,3.5],
    ('11'): [5,1],
    ('12'): [5.5,4],
    ('13'): [6,3.6],
    ('14'): [6.6,2.7],
    ('15'): [7,5.7],
    ('16'): [7.6,4.6],
    ('17'): [8.5,6],
    ('18'): [9,6.8],
    ('19'): [10,7.3]
})

model = gp.Model('CurveFitting')

# Constant term of the function f(x). This is a free continuous variable that can take positive and negative values. 
a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")

# Coefficient of the linear term of the function f(x). This is a free continuous variable that can take positive 
# and negative values.
b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")

# Non-negative continuous variables that capture the positive deviations
u = model.addVars(observations, vtype=GRB.CONTINUOUS, name="u")

# Non-negative continuous variables that capture the negative deviations
v = model.addVars(observations, vtype=GRB.CONTINUOUS, name="v")

# Non-negative continuous variables that capture the value of the maximum deviation
z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

# Deviation constraints

deviations = model.addConstrs( (b*x[i] + a + u[i] - v[i] == y[i] for i in observations), name='deviations')

# Objective function of problem 1

model.setObjective(u.sum('*') + v.sum('*'))

# Verify model formulation

model.write('CurveFitting.lp')

# Run optimization engine

model.optimize()

# Output report

print("\n\n_________________________________________________________________________________")
print(f"The best straight line that minimizes the absolute value of the deviations is:")
print("_________________________________________________________________________________")
print(f"y = {b.x:.4f}x + ({a.x:.4f})")


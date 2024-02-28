#### Purpose:

Example of using the [GUROBI](https://www.gurobi.com/) optimization solver with a Python interface. The specific example illustrates curve fitting and is adopted from [here](https://gurobi.github.io/modeling-examples/curve_fitting/curve_fitting.html). 

#### Contents:

* <code>gurobi_test.py</code>: Python source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>CurveFitting.lp</code>: Model formulation file
* <code>test.out</code>: Output file

#### Python source code:

```python
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
```

#### Model formulation file: <code>CurveFitting.lp</code>

```
\ Model CurveFitting
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7] + u[8] + u[9] + u[10]
   + u[11] + u[12] + u[13] + u[14] + u[15] + u[16] + u[17] + u[18] + u[19]
   + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7] + v[8] + v[9] + v[10]
   + v[11] + v[12] + v[13] + v[14] + v[15] + v[16] + v[17] + v[18] + v[19]
   + 0 z
Subject To
 deviations[1]: a + u[1] - v[1] = 1
 deviations[2]: a + 0.5 b + u[2] - v[2] = 0.9
 deviations[3]: a + b + u[3] - v[3] = 0.7
 deviations[4]: a + 1.5 b + u[4] - v[4] = 1.5
 deviations[5]: a + 1.9 b + u[5] - v[5] = 2
 deviations[6]: a + 2.5 b + u[6] - v[6] = 2.4
 deviations[7]: a + 3 b + u[7] - v[7] = 3.2
 deviations[8]: a + 3.5 b + u[8] - v[8] = 2
 deviations[9]: a + 4 b + u[9] - v[9] = 2.7
 deviations[10]: a + 4.5 b + u[10] - v[10] = 3.5
 deviations[11]: a + 5 b + u[11] - v[11] = 1
 deviations[12]: a + 5.5 b + u[12] - v[12] = 4
 deviations[13]: a + 6 b + u[13] - v[13] = 3.6
 deviations[14]: a + 6.6 b + u[14] - v[14] = 2.7
 deviations[15]: a + 7 b + u[15] - v[15] = 5.7
 deviations[16]: a + 7.6 b + u[16] - v[16] = 4.6
 deviations[17]: a + 8.5 b + u[17] - v[17] = 6
 deviations[18]: a + 9 b + u[18] - v[18] = 6.8
 deviations[19]: a + 10 b + u[19] - v[19] = 7.3
Bounds
 a free
 b free
End
```

#### Example batch-job submission script:

```bash
#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Set up software environment
module load python/3.7.7-fasrc01
module load gurobi/9.0.2-fasrc01
export PYTHONPATH=/n/sw/gurobi902/linux64/lib/python3.7

# Run program
srun -n 1 -c 1 python gurobi_test.py
```

#### Example Output:

```
$ cat test.out
Using license file /opt/gurobi/gurobi.lic
Set parameter TokenServer to value rclic1.rc.fas.harvard.edu
Gurobi Optimizer version 9.0.2 build v9.0.2rc0 (linux64)
Optimize a model with 19 rows, 41 columns and 75 nonzeros
Model fingerprint: 0x8e06b20f
Coefficient statistics:
  Matrix range     [5e-01, 1e+01]
  Objective range  [1e+00, 1e+00]
  Bounds range     [0e+00, 0e+00]
  RHS range        [7e-01, 7e+00]
Presolve removed 0 rows and 1 columns
Presolve time: 0.04s
Presolved: 19 rows, 40 columns, 75 nonzeros

Iteration    Objective       Primal Inf.    Dual Inf.      Time
       0      handle free variables                          0s
      20    1.1466250e+01   0.000000e+00   0.000000e+00      0s

Solved in 20 iterations and 0.04 seconds
Optimal objective  1.146625000e+01


_________________________________________________________________________________
The best straight line that minimizes the absolute value of the deviations is:
_________________________________________________________________________________
y = 0.6375x + (0.5813)

```
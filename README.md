# sppa

Sequential Piecewise Planar Approximation (SPPA) for piecewise linear programming. A convergent MINLP solver for mathematical programming problems. Arxiv preprint: (tba). 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ``sppa``.

```bash
pip install sppa
```

Use of CPLEX in sppa requires installation of the [Python API](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html) of the CPLEX library. 

``sppa`` also requires the python packages ``PuLP`` and ``numpy`` which should be automatically installed with``pip``. 

## Documentation
API documentation available as a PDF file in the repo above. 

## Usage
An example usage of ``sppa`` on a spring design MINLP problem. AMPL model file available [here](http://www.mcs.anl.gov/~leyffer/MacMINLP/problems/spring.mod). (E. Sangren, Trans. ASME, J. Mech. Design 112, 223-229, 1990)
```python
from sppa import SPPA, Var, NlinExpr # equivalently, from sppa import *

# constant definitions
Pload = 300
Pmax = 1000
delm = 6
delw = 1.25
lmax = 14
Dmax = 3
S = 189000
G = 11.5E6
dmin = 0.2
pi = 3.141592654
d = [0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5]
C_upbound = (Dmax-min(d))/min(d)

# Nonlinear function definitions
def f(D, d_index, N):
    return pi * D * d[d_index] ** 2 * (N + 2) / 4
def div_fun(D, d_index):
    return D / d[d_index]
def K_fun(C):
    return (4*C - 1)/(4*C - 4) + 0.615/C
def S_fun(K, D, d_index):
    return 8 * Pmax * K * D / (pi * d[d_index] ** 3)
def del_fun(N, D, d_index):
    return 8 * (N * D**3) / (G * d[d_index] ** 4)
def lmax_fun(N, d_index):
    return 1.05 * (N+2) * d[d_index]
def d_fun(d_index):
    return d[d_index]

# optimization variable definitions
di = Var('di', low_bound=0, up_bound=10, var_type='int', expand=True)
D = Var('D', low_bound=2*dmin, up_bound=Dmax-min(d))
N = Var('N', low_bound=1, up_bound=100, var_type='int')
C = Var('C', low_bound=1.1, up_bound=C_upbound)
K = Var('K', low_bound=K_fun(C_upbound), up_bound=K_fun(1.1))
del_ = Var('del', low_bound=0)

# initialize solver
prob = SPPA('testproblem_spring')

# set objective
prob.set_objective(NlinExpr(f, D, di, N), name='material')

# add equality constraints
prob.add_equality_constraint(C - NlinExpr(div_fun, D, di), 'C_def')
prob.add_equality_constraint(K - NlinExpr(K_fun, C), 'K_def')
prob.add_equality_constraint(del_ - NlinExpr(del_fun, N, D, di))

# add inequality constraints
prob.add_inequality_constraint(S - NlinExpr(S_fun, K, D, di))
prob.add_inequality_constraint(lmax - Pmax*del_ - NlinExpr(lmax_fun, N, di))
prob.add_inequality_constraint(Dmax - D - NlinExpr(d_fun, di))
prob.add_inequality_constraint(delm - Pload*del_)
prob.add_inequality_constraint((Pmax-Pload)*del_ - delw)

# solve problem
prob.compile(initial_n_pieces=4, n_pieces=3, solver='cplex', contract_frac=0.8)
prob.write()
prob.set_termination_criteria(ftol=None, xtol=1E-6, computation_time=None, max_iterations=100)
result = prob.solve()

print('Objective found: ' + str(result.value))
print('Solution found: ' + str(result.solution))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

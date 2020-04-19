#from sppa.Operations import NlinExpr, Var

available_solvers = ['cbc', 'cplex']
CONT_VAR = 1
INT_VAR = 2
BIN_VAR = 3
OBJECTIVE = 1
EQ_CON = 2
IEQ_CON = 3
MIN_N_PIECES = 3
TOL = 1E-6  # This is the tolerance of the LP solver to determine if an obtained solution is out of bounds
FTOL_TERMINATE = 1
XTOL_TERMINATE = 2
COMP_TIME_TERMINATE = 3
MAX_ITER_TERMINATE = 4
VERTEX_EPS_TERMINATE = 5
ERROR_TERMINATE = 6
#NlinExprType = type(NlinExpr(None, [], None))
#VarType = type(Var(None))
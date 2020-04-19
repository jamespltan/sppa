from sppa.constants import available_solvers
import traceback
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import pulp
from sppa.Utilities import *


class MILPsolver:
    def __init__(self, solver_name):
        self.solver_name = solver_name
        self.prob = None
        self.opt_variables = {}
        self.solved_variables = {}

    def solve(self, milp_msg, epgap=0):
        epgap = 0.05 if epgap is None else epgap
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            try:
                if self.solver_name == 'cbc':
                    #self.prob.solve(solvers.PULP_CBC_CMD(msg=milp_msg, maxSeconds=None, fracGap=None))
                    self.prob.solve(pulp.PULP_CBC_CMD(msg=milp_msg, fracGap=epgap))
                if self.solver_name == 'cplex':
                    self.prob.solve(pulp.CPLEX_PY(msg=milp_msg, epgap=epgap))
            except Exception:
                raise Exception('The MILP solver failed. ')
            if self.prob.status != 1 and self.prob.status != -1:
                raise Exception('MILP solution is ' + LpStatus[self.prob.status])
            for variable in self.prob.variables():
                self.solved_variables[variable.name] = variable.varValue
        return self.solved_variables, self.prob.status

    def initialize_problem(self):
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            self.prob = LpProblem("Optimization Problem", LpMinimize)
            self.solved_variables = {}
            self.opt_variables = {}

    def set_objective(self, lin_vars, const=None):
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            self.prob += lpSum(self.unpack_linvars_const(lin_vars, const))

    def add_equality_constraint(self, lin_vars, const=None):
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            self.prob += lpSum(self.unpack_linvars_const(lin_vars, const)) == 0

    def add_inequality_constraint(self, lin_vars, const=None):
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            self.prob += lpSum(self.unpack_linvars_const(lin_vars, const)) >= 0

    def add_variable(self, variable_name, bounds, var_type):
        if self.solver_name == 'cbc' or self.solver_name == 'cplex':
            self.opt_variables[variable_name] = \
                LpVariable(variable_name, bounds[0], bounds[1], var_type_to_str(var_type))

    def unpack_linvars_const(self, lin_vars, const):
        var_list = [coeff * self.opt_variables[lv_name] for lv_name, coeff in lin_vars.items()]
        if const is not None:
            var_list.append(const)
        return var_list

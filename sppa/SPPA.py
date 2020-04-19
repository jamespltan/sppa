from sppa.Utilities import *
from sppa.MILPsolvers import MILPsolver
from sppa.Operations import *
from collections import OrderedDict
from copy import deepcopy
import time


class SPPA:

    def __init__(self, name='Optimization Problem'):
        check_string(name)
        self.problem_name = name
        self.error = None
        self.objective = None
        self.objective_name = 'objective'
        self.equality_constraints = OrderedDict()
        self.inequality_constraints = OrderedDict()
        self.constraints = OrderedDict()
        self.start_time = None
        self.variables = None
        self.solved_variables = None
        self.variables_to_decompose = None
        self.optimization_variables = None
        self.nlinexprs = None
        self.nlinexprs_byid = None
        self.initial_n_pieces = None
        self.n_pieces = MIN_N_PIECES
        self.initial_n_vertices = None
        self.n_vertices = self.n_pieces + 1
        self.solver = None
        self.decomposition_initialized = False
        self.contract_frac = 0.7
        self.min_iterations = 20
        self.max_iterations = 100
        self.ftol = 1E-6
        self.xtol = 1E-6
        self.tol_wait = 5
        self.ftol_waited = None
        self.xtol_waited = None
        self.iteration_number = None
        self.termination_criteria_satisfied = False
        self.exit_flag = None
        self.computation_time = None
        self.initial_ep_gap = None
        self.infeasible_allowed = None
        self.compiled = False

    def compile(self, solver='cbc', n_pieces=MIN_N_PIECES, initial_n_pieces=None, initial_ep_gap=0.0,
                variable_order=None, contract_frac=0.7):
        if self.objective is None:
            raise Exception('Objective function is not set')
        if variable_order is not None and variable_order != 'alpha_num':
            raise Exception('variable_order is not understood')
        if n_pieces is None:
            n_pieces = MIN_N_PIECES
        check_n_pieces(n_pieces)
        check_contract_frac(contract_frac)
        check_ep_gap(initial_ep_gap)
        solver = check_solver_name(solver)
        self.initial_ep_gap = initial_ep_gap
        self.n_pieces = n_pieces
        self.contract_frac = contract_frac
        self.n_vertices = n_pieces + 1
        check_initial_n_pieces(initial_n_pieces, n_pieces)
        self.initial_n_pieces = n_pieces if initial_n_pieces is None else initial_n_pieces
        self.initial_n_vertices = self.initial_n_pieces + 1
        self.solver = MILPsolver(solver)
        self.get_all_variables(variable_order)
        self.get_all_nlinexprs()
        self.constraints = copy.deepcopy(self.equality_constraints)
        self.constraints.update(self.inequality_constraints)
        self.reference_variables()
        #self.initialize_decomposition()
        self.compiled = True

    def solve(self, verbose=1, milp_msg=False):
        self.check_compiled()
        self.start_time = time.time()
        self.iteration_number = 1
        if verbose == 1:
            print('************************************************')
        self.termination_criteria_satisfied = False
        self.ftol_waited, self.xtol_waited = 0, 0
        while not self.termination_criteria_satisfied:
            t1 = time.time()
            if verbose == 1:
                print('Iteration ' + str(self.iteration_number))
            if self.iteration_number == 1 and not self.decomposition_initialized:
                self.initialize_decomposition()
            if self.iteration_number != 1:
                all_solutions_not_viable, variable_vertices_set = self.decompose_variables()
                if all_solutions_not_viable:
                    warnings.warn('Force terminating algorithm now. '
                                  'All variable solutions are not distinguisable by machine epsilon.',
                                  SPPAWarning)
                    self.exit_flag = VERTEX_EPS_TERMINATE
                    break
            self.assemble_decomposed_problem()

            #print(self.variables['chill_QcVC_0_t0'].vertices)
            #print(self.variables['chill_QcVC_1_t0'].vertices)

            ep_gap = self.initial_ep_gap if self.iteration_number == 1 and self.initial_ep_gap is not None else 0.0
            t1d = time.time()
            try:
                self.solved_variables, prob_status = self.solver.solve(epgap=ep_gap, milp_msg=milp_msg)
                if prob_status == -1:
                    if self.iteration_number == 1:
                        raise Exception(
                            'The MILP solution is infeasible. The MINLP problem can be feasible even though '
                            'the decomposed MILP problem isn\'t. Consider setting a larger initial_n_pieces')
                    else:
                        raise Exception(
                            'The MILP solution is infeasible. The solver may have run for too many iterations '
                            'resulting in shrunk bounds being incompatible between variables. Consider setting'
                            'a looser termination criteria, ftol and/or xtol')
            except Exception as error:
                warnings.warn('Force terminating algorithm now due to MILP solver error. ' + str(error), SPPAWarning)
                self.error = error
                self.exit_flag = ERROR_TERMINATE
                break

            round_integer_variables(self.solved_variables, self.variables)

            #print('PV      ' + str(self.solved_variables['PV_Area']))
            #print('gen_0   ' + str(self.solved_variables['gen_type_0'] if self.solved_variables['gen_slot_0'] != 0 else 'unused'))
            #print('gen_1   ' + str(self.solved_variables['gen_type_1'] if self.solved_variables['gen_slot_1'] != 0 else 'unused'))
            #print('chill_0 ' + str(self.solved_variables['chill_type_0'] if self.solved_variables['chill_slot_0'] != 0 else 'unused'))
            #print('chill_1 ' + str(self.solved_variables['chill_type_1'] if self.solved_variables['chill_slot_1'] != 0 else 'unused'))
            #print('batt    ' + str(self.solved_variables['elec_store_type_0'] if self.solved_variables['elec_store_slot_0'] != 0 else 'unused'))
            #print('tes     ' + str(self.solved_variables['therm_store_type_0'] if self.solved_variables['therm_store_slot_0'] != 0 else 'unused'))

            self.compute_result()
            t2 = time.time()
            violated, max_violation, violated_name = constraints_violated(self.constraints)
            max_solution_change = max([abs(v.dvalue) for v in self.variables.values()]) if self.iteration_number != 1 \
                else None
            s_max_solution_change = '{:.3g}'.format(max_solution_change) if max_solution_change is not None else 'None'
            if verbose == 1:
                if max_violation is None:
                    s_max_violation = 'None'
                else:
                    if violated:
                        s_max_violation = '{:.3g}'.format(max_violation) + ' (' + violated_name + ', not satisfied)'
                        #s_max_violation = '{:.3g}'.format(max_violation) + ' (not satisfied)'
                    else:
                        s_max_violation = '{:.3g}'.format(max_violation) + ' (satisfied)'
                print('{:<30}'.format('Objective: ') + '{:.5g}'.format(self.objective.f))
                print('{:<30}'.format('Optimization variables: ') + str(len(self.optimization_variables)))
                print('{:<30}'.format('Decomposition time: ') + time_str(t1d - t1))
                print('{:<30}'.format('MILP time: ') + time_str(t2 - t1d))
                print('{:<30}'.format('Iteration time: ') + time_str(t2 - t1))
                print('{:<30}'.format('Max solution change: ') + s_max_solution_change)
                if violated is not None:
                    print('{:<30}'.format('Max constraint violation: ') + s_max_violation)
                print('************************************************')

            self.determine_termination_criteria_satisfied()
            self.iteration_number += 1

        if self.iteration_number == 1 and self.exit_flag == ERROR_TERMINATE:
            raise Exception('MILP error occured at first iteration. ')
        total_computation_time = time_str(time.time() - self.start_time)
        if verbose == 1:
            print('{:<30}'.format('Total computation time: ') + time_str(time.time() - self.start_time))

        self.iteration_number -= 1
        self.decomposition_initialized = False
        solution = dict([(variable_name, variable.value) for variable_name, variable in self.variables.items()])
        constraints_satisfied = determine_constraints_satisfied(self.constraints)
        result = SolverResult(self.objective.f, solution, self.iteration_number, self.exit_flag, constraints_satisfied,
                              max_violation, total_computation_time)
        return result

    def set_termination_criteria(self, ftol=1E-6, xtol=1E-6, computation_time=None, min_iterations=5, max_iterations=100,
                                 infeasible_allowed=False, con_tol=None, tol_wait=5):
        check_termination_criteria(ftol, xtol, computation_time, max_iterations, infeasible_allowed, min_iterations,
                                   tol_wait, con_tol)
        self.check_compiled()
        self.ftol = ftol
        self.xtol = xtol
        self.tol_wait = tol_wait
        self.computation_time = computation_time
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.infeasible_allowed = infeasible_allowed
        self.objective.xtol = xtol
        self.objective.ftol = ftol
        for constraint in self.constraints.values():
            constraint.xtol = xtol
            constraint.ftol = ftol
            if con_tol is not None:
                constraint.con_tol = con_tol

    def compute_result(self):
        self.objective.compute_expression(self.solved_variables)
        for constraint in self.constraints.values():
            constraint.compute_expression(self.solved_variables)
        for variable in self.variables.values():
            variable.set_value(self.solved_variables[variable.name])

    def determine_termination_criteria_satisfied(self):
        self.termination_criteria_satisfied = False
        # infeasible_allowed
        if self.infeasible_allowed is False:
            if constraints_violated(self.constraints)[0]:
                return
        if self.min_iterations is not None:
            if self.iteration_number < self.min_iterations:
                return
        # ftol
        if self.ftol is not None:
            if self.iteration_number != 1:
                if abs(self.objective.df) < self.ftol:
                    self.ftol_waited += 1
                    if self.ftol_waited == self.tol_wait:
                        self.termination_criteria_satisfied = True
                        self.exit_flag = FTOL_TERMINATE
                        return
                else:
                    self.ftol_waited = 0
        # xtol
        if self.xtol is not None:
            if self.iteration_number != 1:
                xtol_satisfied = True
                for variable in self.variables.values():
                    if abs(variable.dvalue) > self.xtol:
                        xtol_satisfied = False
                        break
                if xtol_satisfied:
                    self.xtol_waited += 1
                    if self.xtol_waited == self.tol_wait:
                        self.termination_criteria_satisfied = True
                        self.exit_flag = XTOL_TERMINATE
                        return
                else:
                    self.xtol_waited = 0
        # computation time
        if self.computation_time is not None:
            if (time.time() - self.start_time) / 60.0 > self.computation_time:
                self.termination_criteria_satisfied = True
                self.exit_flag = COMP_TIME_TERMINATE
                return
        # max iterations
        if self.max_iterations is not None:
            if self.iteration_number == self.max_iterations:
                self.termination_criteria_satisfied = True
                self.exit_flag = MAX_ITER_TERMINATE
                return

    def decompose_variables(self):
        all_solutions_not_viable = True
        variable_vertices_set = []
        for variable in self.variables_to_decompose.values():
            if not variable.expand and variable.var_type != BIN_VAR:
                if variable.var_type == INT_VAR and variable.bounds[1] - variable.bounds[0] + 1 <= self.n_vertices:
                    continue
                variable_solution = self.solved_variables[variable.name]
                index_a, index_b = local_vertex_indices(variable_solution, variable.vertices)
                if index_a is not None:
                    vertices = next_vertices(variable_solution, variable.vertices, variable.var_type,
                                             variable.bounds, self.n_vertices, self.contract_frac,
                                             breakpoint_fun=variable.breakpoint_fun, **variable.breakpoint_kwargs)
                    if vertices_distinguishable(vertices):
                        variable.set_vertices(vertices)
                        variable_vertices_set.append(variable)
                        all_solutions_not_viable = False
        decomposed_nlinexprs = []
        for nlinexpr in self.nlinexprs.values():
            nlinexpr.decompose_expression(decomposed_nlinexprs, self.variables_to_decompose)
        return all_solutions_not_viable, variable_vertices_set

    def assemble_decomposed_problem(self):
        self.check_compiled()
        self.solver.initialize_problem()
        self.initialize_solver_variables()
        self.solver.set_objective(*self.decomposed_linear_expression(self.objective))
        for constraint in self.equality_constraints.values():
            self.solver.add_equality_constraint(*self.decomposed_linear_expression(constraint))
        for constraint in self.inequality_constraints.values():
            self.solver.add_inequality_constraint(*self.decomposed_linear_expression(constraint))

        # sum of variable pieces constraint i.e. x = x_1 + x_2
        for variable_name, variable in self.variables_to_decompose.items():
            lin_vars = {variable_name: 1.0}
            for decomposed_variable in variable.decomposed_vars:
                lin_vars[decomposed_variable.name] = -1.0
            self.solver.add_equality_constraint(lin_vars)

        # sum of binary pieces constraint i.e. mu_x1 + mu_x2 = 1
        for variable_name, variable in self.variables_to_decompose.items():
            lin_vars = {}
            const = -1.0
            for decomposed_bin in variable.decomposed_bins:
                lin_vars[decomposed_bin.name] = 1.0
            self.solver.add_equality_constraint(lin_vars, const)

        # bounds of variable pieces constraint i.e. xa_1 * mu_x1 <= x_1 <= xb_1 * mu_x1
        for variable_name, variable in self.variables_to_decompose.items():
            for decomposed_var, decomposed_bin in zip(variable.decomposed_vars, variable.decomposed_bins):
                # lower bound
                lin_vars = {decomposed_var.name: 1.0}
                lin_vars.update({decomposed_bin.name: -decomposed_var.bounds[0]})
                self.solver.add_inequality_constraint(lin_vars)
                # upper bound
                lin_vars = {decomposed_var.name: -1.0}
                lin_vars.update({decomposed_bin.name: decomposed_var.bounds[1]})
                self.solver.add_inequality_constraint(lin_vars)

        # sum of decomposed_variables constraint i.e. x_1 = x_11 + x_12, x_2 = x_21 + x_22
        for variable_name, variable in self.variables_to_decompose.items():
            for nlinexpr in self.nlinexprs.values():
                if nlinexpr.sharing_nlinexpr_decomposition:
                    continue
                if variable.name in nlinexpr.vars_fun_names:
                    for piece_index, decomposed_variable in enumerate(variable.decomposed_vars):
                        lin_vars = {decomposed_variable.name: 1.0}
                        dimension_index = nlinexpr.vars_fun_names.index(variable.name)
                        var_slice = (dimension_index,) + (slice(None),)*dimension_index + (piece_index,) + \
                                    (slice(None),)*(len(nlinexpr.vars_fun)-dimension_index)
                        nlinexpr_decomposed_vars = nlinexpr.decomposed_vars[var_slice].flatten()
                        lin_vars.update({dv.name: -1.0 for dv in nlinexpr_decomposed_vars})
                        self.solver.add_equality_constraint(lin_vars)

        # sum of decomposed_binaries constraint i.e. mu_x1 = mu_11 + mu_12, mu_x2 = mu_21 + mu_22
        for nlinexpr in self.nlinexprs.values():
            if nlinexpr.sharing_nlinexpr_decomposition:
                continue
            for dimension_index, variable in enumerate(nlinexpr.vars_fun):
                for piece_index, decomposed_bin in enumerate(variable.decomposed_bins):
                    lin_vars = {decomposed_bin.name: 1.0}
                    bin_slice = (slice(None),)*dimension_index + (piece_index,) + \
                                    (slice(None),)*(len(nlinexpr.vars_fun)-dimension_index)
                    nlinexpr_decomposed_bins = nlinexpr.decomposed_bins[bin_slice].flatten()
                    lin_vars.update({db.name: -1 for db in nlinexpr_decomposed_bins})
                    self.solver.add_equality_constraint(lin_vars)

        # bounds of decomposed variables i.e. xa * mu_11 <= x_11 <= xb * mu_11,  ya * mu_11 <= x11 < beta
        for nlinexpr in self.nlinexprs.values():
            if nlinexpr.sharing_nlinexpr_decomposition:
                continue
            for cube_id in nlinexpr.cube_ids:
                for simplex_index_id, simplex_id in enumerate(nlinexpr.simplex_ids):
                    decomposed_bin = nlinexpr.decomposed_bins[cube_id + (simplex_index_id,)]
                    for step, dimension_step in enumerate(simplex_id):
                        decomposed_var = nlinexpr.decomposed_vars[(dimension_step,) + cube_id + (simplex_index_id,)]
                        prev_decomposed_var = nlinexpr.decomposed_vars[(simplex_id[step-1],) + cube_id +
                                                                       (simplex_index_id,)] if step != 0 else None
                        # low_bound
                        lin_vars = {decomposed_var.name: 1.0}
                        lin_vars.update({decomposed_bin.name: -decomposed_var.bounds[0]})
                        self.solver.add_inequality_constraint(lin_vars)

                        # up bound
                        lin_vars = {decomposed_var.name: -1.0}
                        if step == 0:
                            lin_vars.update({decomposed_bin.name: decomposed_var.bounds[1]})
                        else:
                            ya, yb = decomposed_var.bounds
                            xa, xb = prev_decomposed_var.bounds
                            lin_vars.update({decomposed_bin.name: ya - xa*(yb-ya)/(xb-xa)})
                            lin_vars.update({prev_decomposed_var.name: (yb-ya)/(xb-xa)})
                        self.solver.add_inequality_constraint(lin_vars)

    def decomposed_linear_expression(self, expr):
        self.check_compiled()
        lin_vars = {}
        for lin_var_name, lin_var in expr.lin_vars.items():
            check_lin_var_exists(lin_var_name, lin_vars)
            lin_vars[lin_var_name] = lin_var.coeff
        for key, expr_nlinexpr in expr.nlinexprs.items():
            coeff = expr_nlinexpr.coeff
            nlinexpr = self.nlinexprs[key]
            for decomposed_var in nlinexpr.decomposed_vars.flatten():
                if decomposed_var.name in lin_vars:
                    lin_vars[decomposed_var.name] += decomposed_var.coeff * coeff
                else:
                    lin_vars[decomposed_var.name] = decomposed_var.coeff * coeff
            for decomposed_bin in nlinexpr.decomposed_bins.flatten():
                if decomposed_bin.name in lin_vars:
                    lin_vars[decomposed_bin.name] += decomposed_bin.coeff * coeff
                else:
                    lin_vars[decomposed_bin.name] = decomposed_bin.coeff * coeff
        const = expr.const
        return [lin_vars, const]

    def initialize_solver_variables(self):
        self.optimization_variables = {}
        for variable_name, variable in self.variables.items():
            self.optimization_variables[variable_name] = variable
            self.solver.add_variable(variable_name, variable.bounds, variable.var_type)
        for variable_name, variable in self.variables_to_decompose.items():
            for binary_var in variable.decomposed_bins:
                self.optimization_variables[binary_var.name] = binary_var
                self.solver.add_variable(binary_var.name, binary_var.bounds, binary_var.var_type)
            # decomposed variable bounds to be added via constraints
            for decomposed_var in variable.decomposed_vars:
                self.optimization_variables[decomposed_var.name] = decomposed_var
                self.solver.add_variable(decomposed_var.name, [None, None], decomposed_var.var_type)
        # decomposed variable bounds to be added via constraints
        for nlinexpr in self.nlinexprs.values():
            for decomposed_var in nlinexpr.decomposed_vars.flatten():
                self.optimization_variables[decomposed_var.name] = decomposed_var
                self.solver.add_variable(decomposed_var.name, [None, None], decomposed_var.var_type)
            for decomposed_bin in nlinexpr.decomposed_bins.flatten():
                self.optimization_variables[decomposed_bin.name] = decomposed_bin
                self.solver.add_variable(decomposed_bin.name, [None, None], decomposed_bin.var_type)

    # collect variables to be decomposed in variables_to_decompose
    # initialize the vertices of these variables and initialize the corresponding decomposed variables
    # check that there are no name conflicts in the decomposed variables
    def initialize_decomposition(self):
        all_var_names = set(self.variables.keys())
        self.variables_to_decompose = {}
        for nlinexpr in self.nlinexprs.values():
            for variable in nlinexpr.vars_fun:
                if variable.name not in self.variables_to_decompose:
                    self.variables_to_decompose[variable.name] = self.variables[variable.name]
        for variable in self.variables_to_decompose.values():
            if variable.expand:
                if variable.var_type == BIN_VAR or variable.var_type == CONT_VAR:
                    raise Exception('expanded variable is binary or continuous')
                variable.set_vertices(np.arange(variable.bounds[0], variable.bounds[1] + 1, dtype='int'))
            else:
                if variable.var_type == INT_VAR:
                    if variable.bounds[1] - variable.bounds[0] + 1 <= self.initial_n_vertices:
                        variable.set_vertices(np.arange(variable.bounds[0], variable.bounds[1] + 1, dtype='int'))
                    else:
                        if variable.breakpoint_fun is None:
                            vertices = np.array([int(vertex) for vertex in
                                                 np.linspace(*variable.bounds, self.initial_n_vertices)], dtype='int')
                        else:
                            vertices = variable.breakpoint_fun(variable.bounds[0], variable.bounds[1], self.initial_n_vertices,
                                                               **variable.breakpoint_kwargs)
                            check_vertices(vertices, variable.bounds, variable.var_type)
                        vertices = squash_degenerate_vertices(vertices, variable.bounds)
                        variable.set_vertices(vertices)
                if variable.var_type == CONT_VAR:
                    if variable.breakpoint_fun is None:
                        vertices = np.linspace(*variable.bounds, self.initial_n_vertices)
                    else:
                        vertices = variable.breakpoint_fun(variable.bounds[0], variable.bounds[1], self.initial_n_vertices,
                                                           **variable.breakpoint_kwargs)
                        check_vertices(vertices, variable.bounds, variable.var_type)
                    variable.set_vertices(vertices)
                if variable.var_type == BIN_VAR:
                    variable.set_vertices(np.array([0, 1]))
            for decomposed_bin in variable.decomposed_bins:
                if decomposed_bin.name in all_var_names:
                    raise Exception('variable name for a decomposed variable is the same as an existing variable name, '
                                    + decomposed_bin.name + ', please rename variable to resolve the conflict')
                all_var_names.add(decomposed_bin.name)
            for decomposed_var in variable.decomposed_vars:
                if decomposed_var.name in all_var_names:
                    raise Exception('variable name for a decomposed variable is the same as an existing variable name, '
                                    + decomposed_var.name + ', please rename variable to resolve the conflict')
                all_var_names.add(decomposed_var.name)
        decomposed_nlinexprs = []
        for nlinexpr in self.nlinexprs.values():
            nlinexpr.decompose_expression(decomposed_nlinexprs, self.variables_to_decompose)
            if nlinexpr.sharing_nlinexpr_decomposition:
                continue
            for decomposed_var in nlinexpr.decomposed_vars.flatten():
                if decomposed_var.name in all_var_names:
                    raise Exception('variable name for a decomposed variable is the same as an existing variable name, '
                                    + decomposed_var.name + ', please rename variable to resolve the conflict')
                all_var_names.add(decomposed_var.name)
            for decomposed_bin in nlinexpr.decomposed_bins.flatten():
                if decomposed_bin.name in all_var_names:
                    raise Exception('variable name for a decomposed variable is the same as an existing variable name, '
                                    + decomposed_bin.name + ', please rename variable to resolve the conflict')
                all_var_names.add(decomposed_bin.name)
        self.decomposition_initialized = True

    # Add an objective function to the problem
    def set_objective(self, expr, name=None):
        check_expression(expr, Expr, 'objective function')
        expr = deepcopy(expr)
        expr.expression_type = OBJECTIVE
        expr.singular = False
        if name is None:
            name = 'objective'
        if self.objective is not None:
            print('Replaced objective function (' + self.objective_name + '): ' +
                  self.objective.print_expr(return_str=True))
        self.objective = expr
        self.objective_name = name
        self.compiled = False

    # Add a linear constraint to the problem, name must be unique from other constraints
    def add_equality_constraint(self, expr, name=None, con_tol=1E-4):
        check_expression(expr, Expr, 'constraint')
        check_positive_number_or_none(con_tol, 'constraint tolerance')
        expr = deepcopy(expr)
        expr.expression_type = EQ_CON
        expr.singular = False
        expr.con_tol = con_tol
        if name is None:
            name = 'equality constraint ' + str(len(self.equality_constraints))
        if name in self.inequality_constraints:
            raise Exception('a constraint of the same name exists in the inequality constraints. ')
        if name in self.equality_constraints:
            print('Replaced previous constraint (' + name + '): ' + expr.print_expr(return_str=True))
        self.equality_constraints[name] = expr
        self.compiled = False

    # Add a nonlinear constraint to the problem, name must be unique from other constraints
    def add_inequality_constraint(self, expr, name=None, con_tol=1E-4):
        check_expression(expr, Expr, 'constraint')
        check_positive_number_or_none(con_tol, 'constraint tolerance')
        expr = deepcopy(expr)
        expr.expression_type = IEQ_CON
        expr.singular = False
        expr.con_tol = con_tol
        if name is None:
            name = 'inequality constraint ' + str(len(self.inequality_constraints))
        if name in self.equality_constraints:
            raise Exception('a constraint of the same name exists in the equality constraints. ')
        if name in self.inequality_constraints:
            print('Replaced previous constraint (' + name + '): ' + expr.print_expr(return_str=True))
        self.inequality_constraints[name] = expr
        self.compiled = False

    def remove_equality_constraint(self, name):
        check_string(name)
        if name not in self.equality_constraints:
            print('Equality constraint ' + name + ' cannot be found. ')
            return
        del(self.equality_constraints[name])
        self.compiled = False

    def remove_inequality_constraint(self, name):
        check_string(name)
        if name not in self.inequality_constraints:
            print('Inequality constraint ' + name + ' cannot be found. ')
            return
        del(self.inequality_constraints[name])
        self.compiled = False

    def get_all_variables(self, variable_order='alpha_num'):
        all_variables = list()
        all_variables += deepcopy(self.objective.get_variables())
        for constraint in self.equality_constraints.values():
            all_variables += deepcopy(constraint.get_variables())
        for constraint in self.inequality_constraints.values():
            all_variables += deepcopy(constraint.get_variables())
        self.variables = list()
        variable_names = set()
        for i, variable in enumerate(all_variables):
            if variable.name not in variable_names:
                for j in range(i + 1, len(all_variables)):
                    if variable.name == all_variables[j].name:
                        if not check_var_consistency(variable, all_variables[j], throw=False):
                            raise Exception('Same name variables not consistent across problem')
                self.variables.append(variable)
                variable_names.add(variable.name)
        if variable_order == 'alpha_num':
            self.variables = sorted(self.variables, key=lambda x: alphanum_sort(x.name))
        variable_names = [variable.name for variable in self.variables]
        self.variables = dict(zip(variable_names, self.variables))

    def get_all_nlinexprs(self):
        self.nlinexprs = {}
        self.nlinexprs.update(copy.deepcopy(self.objective.nlinexprs))
        for constraint in self.equality_constraints.values():
            self.nlinexprs.update(copy.deepcopy(constraint.nlinexprs))
        for constraint in self.inequality_constraints.values():
            self.nlinexprs.update(copy.deepcopy(constraint.nlinexprs))
        self.nlinexprs_byid = []
        for nlinexpr_id, nlinexpr in enumerate(self.nlinexprs.values()):
            nlinexpr.nlinexpr_id = nlinexpr_id
            self.nlinexprs_byid.append(nlinexpr)

    # make sure varinfos in variables are referenced in vars_fun in nlinexprs
    def reference_variables(self):
        # reference variables in nonlinear expressions
        for nlinexpr in self.nlinexprs.values():
            for i, var_fun in enumerate(nlinexpr.vars_fun):
                nlinexpr.vars_fun[i] = self.variables[var_fun.name]

    def print(self, return_str=False):
        self.check_compiled()
        s = '\n' + self.problem_name + '\n'
        names_lincon = self.equality_constraints.keys()
        names_nlincon = self.inequality_constraints.keys()
        max_l = max([len(lincon) for lincon in names_lincon] + [len(nlincon) for nlincon in names_nlincon] + [0]) + 1
        max_linhead_l = print_constraints(self.equality_constraints, 'equality', max_l, max_head_l=None)
        max_nlinhead_l = print_constraints(self.inequality_constraints, 'inequality', max_l, max_head_l=None)
        max_varhead_l = print_variables_(self.variables, max_head_l=None)
        max_head_l = max([len('minimize ' + self.objective_name), len(self.problem_name),
                          max_linhead_l, max_nlinhead_l, max_varhead_l]) + 1
        s += '*'*(len(self.problem_name)+2) + '\n'
        s += 'minimize ' + self.objective_name + '\n'
        s += '-'*(max([len('minimize ' + self.objective_name), len(self.problem_name)])+2) + '\n'
        if self.objective is None:
            s += '--Not Defined--\n'
        else:
            s += self.objective.print_expr(return_str=True) + '\n'
        s += 'subject to\n'
        s += '-'*max_head_l + '\n'
        s += print_constraints(self.equality_constraints, 'equality', max_l, max_head_l, return_str=True)
        s += print_constraints(self.inequality_constraints, 'inequality', max_l, max_head_l, return_str=True)
        s += print_variables_(self.variables, max_head_l, return_str=True)
        s += '\n'
        if not return_str:
            sys.stdout.write(s)
        else:
            return s

    def print_variables(self):
        self.check_compiled()
        print_variables_(self.variables)

    def write(self, filename=None):
        if filename is None:
            filename = self.problem_name
        filename += '.sppa'
        f = open(filename, 'w')
        s = self.print(return_str=True)
        f.write(s)
        f.close()

    def print_objective(self):
        if self.objective is None:
            print('objective function is not defined')
            return
        print('Objective Function (' + self.objective_name + ')')
        print(self.objective.print_expr(return_str=True))

    def print_linear_constraints(self):
        print_constraints(self.equality_constraints, 'linear')

    def print_nonlinear_constraints(self):
        print_constraints(self.inequality_constraints, 'nonlinear')

    def check_compiled(self):
        if not self.compiled:
            raise Exception('problem has not been compiled')

    def __str__(self):
        return self.print(return_str=True)


class SolverResult:
    def __init__(self, value, solution, iterations, exit_flag, constraints_satisfied, max_violation,
                 total_computation_time):
        self.value = value
        self.solution = solution
        self.iterations = iterations
        self.exit_flag = exit_flag
        self.constraints_satisfied = constraints_satisfied
        self.max_violation = max_violation
        self.total_computation_time = total_computation_time
        self.message = None
        self.generate_message()

    def generate_message(self):
        if self.exit_flag == FTOL_TERMINATE:
            self.message = 'Solver terminated because change in objective is smaller than ftol. '
        if self.exit_flag == XTOL_TERMINATE:
            self.message = 'Solver terminated because maximum change in solution variables is smaller than xtol. '
        if self.exit_flag == COMP_TIME_TERMINATE:
            self.message = 'Solver terminated because computation time is exceeded. '
        if self.exit_flag == MAX_ITER_TERMINATE:
            self.message = 'Solver terminated because maximum iterations has been reached. '
        if self.exit_flag == VERTEX_EPS_TERMINATE:
            self.message = 'Solver terminated because variable solutions not distinguishable by milp eps'
        if self.exit_flag == ERROR_TERMINATE:
            self.message = 'Solver terminated because of an error with the MILP solver'
        if self.constraints_satisfied:
            self.message += '\nAll constraints satisfied within constraint tolerance. '
        else:
            self.message += '\nNot all constraints satisfied within constraint tolerance. '

    def print_message(self):
        print(self.message)

    def __str__(self):
        return '\n' + self.message + '\n'

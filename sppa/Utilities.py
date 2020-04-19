from collections import Counter, Callable, Hashable
import itertools
import warnings
import numbers
import numpy as np
from sppa.constants import *
import sys
import re
from copy import copy
from scipy.optimize import newton


class SPPAWarning(Warning):
    pass


class SPPAIllegalArgumentError(ValueError):
    pass


def alphanum_sort(s):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key.lower())]
    return sorted(s, key=alphanum_key)


def num_str(num, started=False):
    if num is None:
        return 'None'
    if num < 0:
        if started is False:
            return '-' + '{:.5g}'.format(-num)
        else:
            return ' - ' + '{:.5g}'.format(-num)
    else:
        return '{:.6g}'.format(num)


def time_str(x):
    x, seconds = divmod(x, 60)
    x, minutes = divmod(x, 60)
    days, hours = divmod(x, 24)
    s = ''
    if days != 0:
        if days == 1:
            s += str(int(days)) + ' day, '
        else:
            s += str(int(days)) + ' days, '
    if hours != 0:
        if hours == 1:
            s += str(int(hours)) + ' hour, '
        else:
            s += str(int(hours)) + ' hours, '
    if minutes != 0:
        if minutes == 1:
            s += str(int(minutes)) + ' minute, '
        else:
            s += str(int(minutes)) + ' minutes, '
    s += '{:.1f}'.format(seconds) + ' seconds'
    return s


def var_type_to_str(var_type):
    if var_type == CONT_VAR:
        return 'Continuous'
    if var_type == INT_VAR:
        return 'Integer'
    if var_type == BIN_VAR:
        return 'Binary'


def round_integer_variables(solved_variables, variables):
    for var_name, value in solved_variables.items():
        if var_name in variables and \
                (variables[var_name].var_type == INT_VAR or variables[var_name].var_type == BIN_VAR):
            solved_variables[var_name] = int(np.round(value))


def next_vertices(solution, original_vertices, var_type, bounds, n_vertices, contract_frac=0.7, breakpoint_fun=None,
                  **breakpoint_kwargs):
    width = (original_vertices[-1] - original_vertices[0]) * contract_frac
    #contract_width = (original_vertices[-1] - original_vertices[0]) * contract_frac
    #expand_width = np.mean(original_vertices[1:] - original_vertices[:-1]) * expand_frac
    #width = min(contract_width, expand_width)
    a = max(bounds[0], solution - width / 2)
    b = min(bounds[1], solution + width / 2)
    if a == bounds[0]:
        b = bounds[0] + width
    if b == bounds[1]:
        a = bounds[1] - width
    if breakpoint_fun is not None:
        vertices = breakpoint_fun(a, b, n_vertices, **breakpoint_kwargs)
        check_vertices(vertices, bounds, var_type)
    else:
        vertices = np.linspace(a, b, n_vertices)
    if var_type == INT_VAR:
        vertices = np.array([int(vertex) for vertex in vertices])
        vertices = squash_degenerate_vertices(vertices, bounds)
    return vertices


def next_vertices__(solution, original_vertices, index_a, index_b, var_type, bounds, n_vertices, expand_multiple=1.5):
    vertices = np.copy(original_vertices)
    h = np.mean(vertices[1:] - vertices[:-1])
    if index_a == index_b and (index_a == 0 or index_a == len(vertices) - 1):
        if index_a == 0:
            expanded_vertex = original_vertices[1] - (original_vertices[-1] - original_vertices[0])*expand_multiple
            vertices = np.linspace(max(bounds[0], expanded_vertex), original_vertices[1], n_vertices)
        if index_a == len(vertices) - 1:
            expanded_vertex = original_vertices[-2] + (original_vertices[-1] - original_vertices[0])*expand_multiple
            vertices = np.linspace(original_vertices[-2], min(bounds[1], expanded_vertex), n_vertices)
    else:
        vertices = np.linspace(max(bounds[0], solution-h), min(bounds[1], solution+h), n_vertices)
    if var_type == INT_VAR:
        vertices = np.array([int(vertex) for vertex in vertices])
        vertices = squash_degenerate_vertices(vertices, bounds)
    return vertices


def next_vertices_(original_vertices, index_a, index_b, var_type, bounds, keep_bounds=False):
    # if keep_bounds is True, 5 pieces i.e. 6 vertices are needed
    vertices = np.copy(original_vertices)
    if keep_bounds is False:
        if index_a == index_b:
            if index_a == 0:
                reflected_vertex = original_vertices[0] - (original_vertices[1] - original_vertices[0])
                vertices = np.linspace(max(bounds[0], reflected_vertex), original_vertices[1], len(vertices))
            if index_a == len(vertices) - 1:
                reflected_vertex = original_vertices[-1] + (original_vertices[-1] - original_vertices[-2])
                vertices = np.linspace(original_vertices[-2], min(bounds[1], reflected_vertex), len(vertices))
            if index_a != 0 and index_a != len(vertices) - 1:
                vertices = np.linspace(original_vertices[index_a-1], original_vertices[index_a+1], len(vertices))
        else:
            if index_a == 0:
                reflected_vertex = original_vertices[0] - (original_vertices[1] - original_vertices[0])
                vertices = np.linspace(max(bounds[0], reflected_vertex), original_vertices[1], len(vertices))
            if index_a == len(vertices) - 2:
                reflected_vertex = original_vertices[-1] + (original_vertices[-1] - original_vertices[-2])
                vertices = np.linspace(original_vertices[-2], min(bounds[1], reflected_vertex), len(vertices))
            if index_a != 0 and index_a != len(vertices) - 2:
                vertices = np.linspace(original_vertices[index_a], original_vertices[index_b], len(vertices))
    else:
        if index_a == index_b:
            if 1 < index_a < len(vertices) - 2:
                vertices[1] = original_vertices[index_a - 1]
                vertices[-2] = original_vertices[index_a + 1]
                vertices[1:-1] = np.linspace(vertices[1], vertices[-2], len(vertices) - 2)
            if index_a == 0 or index_a == 1:
                if index_a == 0:
                    vertices[-2] = original_vertices[1]
                if index_a == 1:
                    vertices[-2] = original_vertices[2]
                vertices[0:-1] = np.linspace(vertices[0], vertices[-2], len(vertices) - 1)
            if index_a == len(vertices) - 1 or index_a == len(vertices) - 2:
                if index_a == len(vertices) - 1:
                    vertices[1] = original_vertices[-2]
                if index_a == len(vertices) - 2:
                    vertices[1] = original_vertices[-3]
                vertices[1:] = np.linspace(vertices[1], vertices[-1], len(vertices) - 1)
        if index_a != index_b:
            if 0 < index_a < len(vertices) - 2:
                vertices[1] = original_vertices[index_a]
                vertices[-2] = original_vertices[index_b]
                vertices[1:-1] = np.linspace(vertices[1], vertices[-2], len(vertices) - 2)
            if index_a == 0:
                vertices[-2] = original_vertices[1]
                vertices[0:-1] = np.linspace(vertices[0], vertices[-2], len(vertices) - 1)
            if index_a == len(vertices) - 2:
                vertices[1] = original_vertices[-2]
                vertices[1:] = np.linspace(vertices[1], vertices[-1], len(vertices) - 1)
    if var_type == INT_VAR:
        vertices = np.array([int(vertex) for vertex in vertices])
        vertices = squash_degenerate_vertices(vertices, bounds)
    return vertices


def squash_degenerate_vertices(vertices, bounds):
    c = Counter(vertices)
    v = np.zeros(bounds[1]-bounds[0]+1, dtype='int')
    vertex_a = bounds[0]
    v[np.array(list(c.keys())) - vertex_a] = list(c.values())
    excess = sum(v[v > 1] - 1)
    if excess == 0:
        return vertices
    md_i = int(np.median(vertices) - vertex_a)
    if v[md_i] == 0:
        v[md_i] = 1
        excess -= 1
    for i in range(1, max(md_i, len(v)-1-md_i)):
        if md_i - i >= 0 and v[md_i - i] == 0:
            v[md_i - i] = 1
            excess -= 1
        if excess == 0:
            break
        if md_i + i <= len(v) - 1 and v[md_i + i] == 0:
            v[md_i + i] = 1
            excess -= 1
        if excess == 0:
            break
    vertices = np.arange(vertex_a, bounds[1]+1)
    v = v.astype('bool')
    vertices = vertices[v]
    return vertices


def local_vertex_indices(value, vertices):
    boundary_divisor = 100.0
    for i, vertex in enumerate(vertices):
        if value == vertex:
            return i, i
        """
        if abs(value-vertex) < TOL:
            if i != len(vertices) - 1 and abs(value-vertices[i+1]) < TOL:
                return i, i+1
            else:
                return i, i
        """
        if vertex > value:
            if i == 0:
                if vertex - value > TOL:
                    break
                else:
                    return 0, 0
            if value - vertices[i-1] < (vertex - vertices[i-1])/boundary_divisor:
                return i-1, i-1
            if vertex - value < (vertex - vertices[i-1])/boundary_divisor:
                return i, i
            return i-1, i
        if i == len(vertices) - 1:
            if value - vertex < TOL:
                return i, i
    return None, None
    # raise ValueError('solution obtained is out of vertices bounds')


# if out-of-bounds, will use minimum and maximum bounds instead of raising exception
def retrieve_variable_value_lenient(variables_values, var_info):
    try:
        x = variables_values[var_info.name]
    except KeyError:
        raise Exception('unable to access variable_name in variables_values')
    if var_info.var_type == INT_VAR:
        if isinstance(x, numbers.Number) and not isinstance(x, numbers.Integral):
            x = int(np.round(x))
        if not isinstance(x, numbers.Integral):
            raise TypeError('variable_value is not an integer for integer var_type')
    if var_info.var_type == BIN_VAR:
        if abs(x) < TOL:
            x = 0
        if abs(x-1) < TOL:
            x = 1
        x = int(np.round(x))
        if x != 0 and x != 1:
            raise ValueError('variable value is not 0 nor 1 for binary var_type')
    if var_info.var_type == CONT_VAR:
        if not isinstance(x, numbers.Number):
            raise TypeError('variable value is not a number for continuous var_type')
    if var_info.bounds[0] is not None:
        if x < var_info.bounds[0]:
            x = var_info.bounds[0]
    if var_info.bounds[1] is not None:
        if x > var_info.bounds[1]:
            x = var_info.bounds[1]
    return x


def retrieve_variable_value(variables_values, var_info):
    try:
        x = variables_values[var_info.name]
    except KeyError:
        raise Exception('unable to access variable_name in variables_values')
    if var_info.var_type == INT_VAR:
        if isinstance(x, numbers.Number) and not isinstance(x, numbers.Integral):
            if abs(np.floor(x) - x) > TOL and abs(np.ceil(x) - x) > TOL:
                raise ValueError('variable value of ' + str(var_info.name) + ' is not an integer for integer var_type')
            x = int(np.round(x))
        if not isinstance(x, numbers.Integral):
            raise TypeError('variable value of ' + str(var_info.name) + ' is not an integer for integer var_type')
    if var_info.var_type == BIN_VAR:
        if abs(x) < TOL:
            x = 0
        if abs(x-1) < TOL:
            x = 1
        if x != 0 and x != 1:
            raise ValueError('variable value is not 0 nor 1 for binary var_type')
    if var_info.var_type == CONT_VAR:
        if not isinstance(x, numbers.Number):
            raise TypeError('variable value is not a number for continuous var_type')
    if var_info.bounds[0] is not None:
        if x + TOL < var_info.bounds[0]:
            raise ValueError('variable value is smaller than lower bound')
        if x < var_info.bounds[0]:
            x = var_info.bounds[0]
    if var_info.bounds[1] is not None:
        if x - TOL > var_info.bounds[1]:
            raise ValueError('variable value is larger than upper bound')
        if x > var_info.bounds[1]:
            x = var_info.bounds[1]
    return x


def generate_decomposed_var_array(n_pieces_all, depth=None):
    if depth is None:
        depth = 0
    if depth == len(n_pieces_all) - 1:
        return [[]] * n_pieces_all[-1]
    arr = []
    for i in range(n_pieces_all[depth]):
        arr.append(generate_decomposed_var_array(n_pieces_all, depth+1))
    return arr


def constraints_violated(constraints):
    """
    :param constraints: constraints to be evaluated
    :return [0]: True if there are any constraints that are violated, false otherwise
    :return [1]: Maximum violation if one or more constraints exist, else None
    :return [2]: Name of the maximum constraint violated if one or more constraints exist, else None
    """
    n = []
    v = []
    violated = False
    for constraint_name, constraint in constraints.items():
        n.append(constraint_name)
        v.append(constraint.violation)
        if constraint.violated:
            violated = True
    if not v:
        return False, None, None
    return violated, max(v), n[v.index(max(v))]


def print_variables_(variables, max_head_l=32, return_str=False):
    variables = variables.values()
    return_max_head_l = False
    if max_head_l is None:
        return_max_head_l = True
        max_head_l = 0
    lb_l, var_l = zip(*[[len(num_str(variable.bounds[0])), len(variable.name)] if variable.var_type != BIN_VAR
                        else [0, 0] for variable in variables])
    max_lb_l, max_var_l = list(map(max, [lb_l, var_l]))
    max_var_l += 2
    s = ['Continuous Variables']
    s += ['-' * max_head_l]
    printed = False
    for variable in variables:
        if variable.var_type == CONT_VAR:
            s += [('{:^' + str(max_lb_l) + '}').format(num_str(variable.bounds[0])) +
                  '  <=  ' + ('{:^' + str(max_var_l) + '}').format(variable.name) + '  <=  ' +
                  num_str(variable.bounds[1])]
            printed = True
    if not printed:
        s += ['---No continuous variables---']
    s += ['-' * max_head_l]
    s += ['Integer Variables']
    s += ['-' * max_head_l]
    printed = False
    for variable in variables:
        if variable.var_type == INT_VAR:
            if variable.expand:
                brac1 = '<'
                brac2 = '>'
            else:
                brac1 = ''
                brac2 = ''
            s += [('{:^' + str(max_lb_l) + '}').format(num_str(variable.bounds[0])) +
                  '  <=  ' + ('{:^' + str(max_var_l) + '}').format(brac1 + variable.name + brac2) + '  <=  ' +
                  num_str(variable.bounds[1])]
            printed = True
    if not printed:
        s += ['---No integer variables---']
    s += ['-' * max_head_l]
    s += ['Binary Variables']
    s += ['-' * max_head_l]
    printed = False
    for variable in variables:
        if variable.var_type == BIN_VAR:
            s += [variable.name]
            printed = True
    if not printed:
        s += ['---No binary variables---']
    s += ['-' * max_head_l]
    if return_max_head_l:
        max_head_l = max([len(s_) for s_ in s])
        return max_head_l
    if not return_str:
        sys.stdout.write('\n'.join(s) + '\n')
    else:
        return '\n'.join(s)


def print_constraints(constraints, con_type, max_l=None, max_head_l=32, return_str=False):
    return_max_head_l = False
    if max_head_l is None:
        return_max_head_l = True
        max_head_l = 0
    s = []
    if con_type != 'equality' and con_type != 'inequality':
        raise Exception('type of constraint not understood')
    if con_type == 'equality':
        s += ['Equality Constraints']
    if con_type == 'inequality':
        s += ['Inequality Constraints (>=0)']
    s += ['-' * max_head_l]
    names = constraints.keys()
    if max_l is None:
        max_l = max([len(name) for name in names]) + 1
    if not names:
        s += ['---No constraints---']
    for name in names:
        s += [('{:<' + str(max_l) + '}').format(name) + ':    ' +
              constraints[name].print_expr(return_str=True)]
    s += ['-' * max_head_l]
    if return_max_head_l:
        max_head_l = max([len(s_) for s_ in s])
        return max_head_l
    if not return_str:
        sys.stdout.write('\n'.join(s) + '\n')
    else:
        return '\n'.join(s) + '\n'


def vertices_distinguishable(vertices):
    for i in range(len(vertices)-1):
        if vertices[i+1] == vertices[i] or vertices[i+1] < vertices[i] or vertices[i+1] < vertices[i] + TOL:
            return False
    return True


def determine_constraints_satisfied(constraints):
    constraints_satisfied = True
    for constraint in constraints.values():
        if constraint.violated:
            constraints_satisfied = False
            break
    return constraints_satisfied


def check_termination_criteria(ftol, xtol, computation_time, max_iterations, infeasible_allowed, min_iterations,
                               tol_wait, con_tol):
    check_boolean(infeasible_allowed, 'infeasible_allowed')
    check_positive_number_or_none(ftol, 'ftol')
    check_positive_number_or_none(xtol, 'xtol')
    check_positive_number_or_none(con_tol, 'con_tol')
    check_positive_number_or_none(computation_time, 'computation_time')
    check_positive_integer_or_none(max_iterations, 'max_iterations')
    check_positive_integer_or_none(min_iterations, 'min_iterations')
    check_positive_integer_or_none(tol_wait, 'tol_wait')
    if max_iterations < min_iterations:
        raise Exception('min_iterations greater than max_iterations')
    if ftol is None and xtol is None and computation_time is None and max_iterations is None:
        raise Exception('no termination criteria defined')
    if computation_time is not None:
        warnings.warn('Warning, computation will only stop between iterations if computation_time is exceeded',
                      SPPAWarning)


def check_bounds(low_bound, up_bound, var_type, expand):
    check_number_or_none(up_bound, 'upper bound')
    check_number_or_none(low_bound, 'lower bound')
    if up_bound is not None and low_bound is not None:
        if up_bound < low_bound:
            raise Exception('lower bound is larger than upper bound')
        if up_bound == low_bound:
            raise Exception('lower bound equal to larger bound')
    if var_type == BIN_VAR:
        if low_bound is not None and low_bound != 0 or \
                up_bound is not None and up_bound != 1:
            raise Exception('bounds for binary variable are not None and not [0, 1]')
        if expand is True:
            warnings.warn('Warning, binary variables are automatically expanded by definition', SPPAWarning)
            expand = False
        return [None, None], expand
    if var_type == INT_VAR:
        if low_bound is not None and not isinstance(low_bound, numbers.Integral) or \
                up_bound is not None and not isinstance(up_bound, numbers.Integral):
            raise Exception('bounds for integer variable are not None and not integers')
    if expand:
        if low_bound is None or up_bound is None:
            raise Exception('non-binary expanded variables must have bounds defined')
        if var_type != INT_VAR:
            raise Exception('expanded variable must be an integer variable')
    if var_type == INT_VAR:
        return [int(low_bound), int(up_bound)], expand
    else:
        return [low_bound, up_bound], expand


def check_vartype(vartype):
    if not isinstance(vartype, str) or not isinstance(vartype, str) or not isinstance(vartype, str):
        raise Exception('variable type is not a string')
    vartype = vartype.lower()
    if vartype == 'cont' or vartype == 'continuous':
        return CONT_VAR
    if vartype == 'int' or vartype == 'integer':
        return INT_VAR
    if vartype == 'bin' or vartype == 'binary':
        return BIN_VAR
    raise Exception('variable type is not understood')


def check_solver_name(solver_name):
    if not isinstance(solver_name, str):
        raise TypeError(solver_name + ' is not a string')
    solver_name = solver_name.lower()
    if solver_name not in available_solvers:
        raise Exception('invalid solver_name')
    return solver_name


def check_variables(variables, Var):
    if not check_iterable(variables) and not isinstance(variables, Var):
        raise Exception('variables are not iterable or variable is not a Var object')
    if isinstance(variables, Var):
        variables = [variables]
    for i, variable in enumerate(variables):
        if not isinstance(variable, Var):
            raise Exception('variable(s) in a nonlinear expression is not a Var object')
        if not variable.singular:
            raise Exception('variable(s) in a nonlinear expression is not singular')
        if variable.var_type != BIN_VAR and None in variable.bounds:
            raise Exception('variable(s) in nonlinear expressions must have bounds')
        for variable_ in variables[i+1:]:
            if variable.name == variable_.name:
                check_var_consistency(variable, variable_)
    return variables


def check_vertices(vertices, bounds, var_type):
    if not check_iterable(vertices):
        raise Exception('vertices is not iterable')
    if var_type == BIN_VAR:
        if list(vertices) != [0, 1]:
            raise ValueError('vertices of a binary variable not [0, 1]')
    else:
        for vertex in vertices:
            if not isinstance(vertex, numbers.Number):
                raise Exception('vertex is not a number')
            if vertex > bounds[1] or vertex < bounds[0]:
                raise Exception('invalid vertex is out of bounds')


def check_extended_variables(vars_fun, vars_exp, Var):
    if not check_iterable(vars_exp) and not isinstance(vars_exp, Var):
        raise Exception('Extended variables are not iterable or variable is not a Var object')
    if isinstance(vars_exp, Var):
        vars_exp = {vars_exp}
    for var_exp in vars_exp:
        if not var_exp.singular:
            raise Exception('variable(s) in a nonlinear expression is not singular')
        if var_exp.var_type != BIN_VAR and var_exp.var_type != INT_VAR:
            raise Exception('expanded variable is not a discrete variable')
        if var_exp.var_type == INT_VAR and None in var_exp.bounds:
            raise Exception('expanded variable must have bounds defined')
        var_exists = False
        for var_fun in vars_fun:
            if var_exp.name == var_fun.name:
                if not check_var_consistency(var_exp, var_fun):
                    raise Exception('same name but inconsistent variables in vars_fun and vars_exp')
                var_exists = True
        if not var_exists:
            raise Exception('var_exp not found in var_fun')
    return set(vars_exp)


def check_var_consistency(var1, var2, throw=True):
    if var1.bounds != var2.bounds:
        if throw:
            raise Exception('variable bounds of ' + var1.name + ' are not the same when comparing expressions')
        else:
            return False
    if var1.var_type != var2.var_type:
        if throw:
            raise Exception('variable type of ' + var1.name + ' is not the same when comparing expressions')
        else:
            return False
    if var1.name != var2.name:
        if throw:
            raise Exception('variable names are not the same when comparing expressions')
        else:
            return False
    if var1.expand != var2.expand:
        if throw:
            raise Exception('variable expanded truth values for ' + var1.name + ' are not the same '
                                                                                'when comparing expressions')
        else:
            return False
    if var1.breakpoint_fun != var2.breakpoint_fun:
        if throw:
            raise Exception('breakpoint function for ' + var1.name + ' are not the same when comparing expressions')
        else:
            return False
    if var1.breakpoint_kwargs != var2.breakpoint_kwargs:
        if throw:
            raise Exception('breakpoint kwargs for ' + var1.name + ' are not the same when comparing expressions')
        else:
            return False
    return True


def check_lin_var_exists(lin_var_name, lin_vars):
    if lin_var_name in lin_vars:
        raise Exception('lin_var_name already exists in lin_vars, potentially leading to conflict in coefficients')


def check_vars_exp_consistency(vars_exp_1, vars_exp_2):
    if len(vars_exp_1) != len(vars_exp_2):
        raise Exception('expanded variables are not the same in an operation combining or comparing two nonlinear '
                        'expressions with the same variables')
    for var1 in vars_exp_1:
        consistent = False
        for var2 in vars_exp_2:
            if var1.name == var2.name:
                check_var_consistency(var1, var2)
                consistent = True
        if not consistent:
            raise ('expanded variables are not the same in an operation combining or comparing two nonlinear '
                   'expressions with the same variables')


def check_positive_number(x, s):
    if not isinstance(x, numbers.Number):
        raise TypeError(s + ' is not a number')
    if x <= 0:
        raise ValueError(s + ' is not positive')


def check_positive_number_or_none(x, s):
    check_number_or_none(x, s)
    if x is not None and x <= 0:
        raise ValueError(s + ' is smaller than or equal to zero')


def check_number_or_none(x, s):
    if not isinstance(x, numbers.Number) and x is not None:
        raise TypeError(s + ' is not a number nor None')


def check_positive_integer_or_none(x, s):
    if not isinstance(x, numbers.Integral) and x is not None:
        raise TypeError(s + ' is not an integer nor None')
    if x is not None and x <= 0:
        raise ValueError(s + ' is smaller than or equal to zero')


def check_string(s):
    if not isinstance(s, str):
        raise TypeError('name is not a string')


def check_expression(expr, Expr, statement='statement'):
    if not isinstance(expr, Expr):
        raise TypeError(statement + ' is not an expression')
    if not expr.lin_vars and not expr.nlinexprs:
        raise SPPAIllegalArgumentError(statement + ' does not contain any linear variables nor nonlinear expressions')


def check_coeff_number(coeff):
    if not isinstance(coeff, numbers.Number):
        raise TypeError('coefficient is not a number')


def check_iterable(x):
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def check_boolean(x, name):
    if not isinstance(x, bool):
        raise TypeError(name + ' is not a boolean')


def check_dependent(dependent, var_type, expand, var_name):
    check_boolean(dependent, 'dependent')
    if dependent:
        if var_type == INT_VAR:
            if not expand:
                raise ValueError('dependent integer variables must be expanded')
        if var_type == CONT_VAR:
            warnings.warn('A continuous variable ' + var_name + ' that is dependent will likely lead to an inaccurate '
                                                                'estimation of any NlinExpr that it is a function of',
                          SPPAWarning)


def check_ep_gap(ep_gap):
    if not isinstance(ep_gap, numbers.Number) and ep_gap is not None:
        raise TypeError('ep_gap is not a number or None')
    if ep_gap is not None and ep_gap < 0:
        raise ValueError('ep_gap is smaller than zero')


def check_n_pieces(n_pieces):
    if not isinstance(n_pieces, numbers.Integral) or n_pieces < MIN_N_PIECES:
        raise ValueError('n_pieces must be >= ' + str(MIN_N_PIECES))


def check_contract_frac(contract_frac):
    if not isinstance(contract_frac, numbers.Number):
        raise TypeError('contract_frac must be a number')
    if 0.99 < contract_frac or contract_frac < 0.1:
        raise ValueError('contract frac must be between 0.1 and 0.99')


def check_initial_n_pieces(initial_n_pieces, n_pieces):
    check_n_pieces(n_pieces)
    if initial_n_pieces is not None and (not isinstance(initial_n_pieces, numbers.Integral)
                                         or initial_n_pieces < n_pieces):
        raise ValueError('initial_n_pieces must be >= n_pieces or None')


def check_kwargs(kwargs):
    for k, v in kwargs.items():
        if not isinstance(v, Hashable):
            raise TypeError('kwargs is not hashable')


def check_fun(fun):
    if not isinstance(fun, Callable):
        raise TypeError('not a function')


def check_breakpoint(fun, kwargs):
    if fun is None and kwargs:
        raise TypeError('breakpoint function is not defined by its kwargs are')
    check_kwargs(kwargs)
    if fun is not None:
        check_fun(fun)

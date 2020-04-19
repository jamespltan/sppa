from sppa.Utilities import *
import copy
from collections import OrderedDict, Callable


class Expr:
    def __init__(self):
        self.singular = False
        self.nlinexprs = dict()
        self.lin_vars = dict()
        self.bounds = None
        self.var_type = None
        self.expression_type = None
        self.const = None
        self.name = None
        self.df = None
        self.f = None
        self.violated = None
        self.violation = None
        self.ftol = None
        self.xtol = None
        self.con_tol = None

    def __add__(self, expr):
        return add_expr(self, expr)

    def __radd__(self, expr):
        return add_expr(self, expr)

    def __iadd__(self, expr):
        return add_expr(self, expr, return_copy=False)

    def __mul__(self, coeff):
        return mul_expr(self, coeff)

    def __rmul__(self, coeff):
        return mul_expr(self, coeff)

    def __imul__(self, coeff):
        return mul_expr(self, coeff, return_copy=False)

    def __truediv__(self, coeff):
        return div_expr(self, coeff)

    def __rtruediv__(self, expr):
        raise Exception('can\'t take the reciprocal of an expression')

    def __itruediv__(self, coeff):
        return div_expr(self, coeff, return_copy=False)

    def __sub__(self, expr):
        return sub_expr(self, expr)

    def __rsub__(self, expr):
        return rsub_expr(self, expr)

    def __isub__(self, expr):
        return sub_expr(self, expr, return_copy=False)

    def __pos__(self):
        return self

    def __neg__(self):
        return mul_expr(self, -1)

    def __str__(self):
        return self.print_expr(return_str=True)

    def compute_expression(self, variables_values):
        f, self.violation, self.violated = self.evaluate_expression_lenient(variables_values, return_violation=True)
        if self.f is not None:
            self.df = f - self.f
        self.f = f

    def evaluate_expression(self, variables_values, return_violation=False):
        f = 0 if self.const is None else self.const
        for variable_name, var_info in self.lin_vars.items():
            x = retrieve_variable_value(variables_values, var_info)
            f = f + var_info.coeff * x
        for nlinexpr in self.nlinexprs.values():
            args = []
            for var_fun in nlinexpr.vars_fun:
                args += [retrieve_variable_value(variables_values, var_fun)]
            f = f + nlinexpr.coeff * nlinexpr.fun_nlin(*args, **nlinexpr.kwargs)
        violation = None
        violated = None
        if self.expression_type == EQ_CON and self.con_tol is not None:
            violation = abs(f)
            if abs(f) > self.con_tol:
                violated = True
            else:
                violated = False
        if self.expression_type == IEQ_CON and self.con_tol is not None:
            violation = abs(min(0, f))
            if f < -self.con_tol:
                violated = True
            else:
                violated = False
        if return_violation:
            return f, violation, violated
        else:
            return f

    def evaluate_expression_lenient(self, variables_values, return_violation=False):
        f = 0 if self.const is None else self.const
        for variable_name, var_info in self.lin_vars.items():
            x = retrieve_variable_value_lenient(variables_values, var_info)
            f = f + var_info.coeff * x
        for nlinexpr in self.nlinexprs.values():
            args = []
            for var_fun in nlinexpr.vars_fun:
                args += [retrieve_variable_value_lenient(variables_values, var_fun)]
            f = f + nlinexpr.coeff * nlinexpr.fun_nlin(*args, **nlinexpr.kwargs)
        violation = None
        violated = None
        if self.expression_type == EQ_CON and self.con_tol is not None:
            violation = abs(f)
            if abs(f) > self.con_tol:
                violated = True
            else:
                violated = False
        if self.expression_type == IEQ_CON and self.con_tol is not None:
            violation = abs(min(0, f))
            if f < -self.con_tol:
                violated = True
            else:
                violated = False
        if return_violation:
            return f, violation, violated
        else:
            return f

    def get_variables(self):
        variables = []
        for variable in self.lin_vars.values():
            variables.append(variable)
        for nlinexpr in self.nlinexprs.values():
            for variable in nlinexpr.vars_fun:
                variables.append(variable)
        return variables

    def print_expr(self, return_str=False):
        started = False
        s = ''
        for var_name, var_info in self.lin_vars.items():
            if started and var_info.coeff > 0:
                s += ' + '
            if var_info.coeff != 1.0:
                if var_info.coeff == -1:
                    if not started:
                        s += '-'
                    else:
                        s += ' - '
                else:
                    s += num_str(var_info.coeff, started) + '*'
            if not started:
                started = True
            s += var_name
        for nlinexpr in self.nlinexprs.values():
            if started and nlinexpr.coeff > 0:
                s += ' + '
            if nlinexpr.coeff != 1.0:
                if nlinexpr.coeff == -1:
                    if not started:
                        s += '-'
                    else:
                        s += ' - '
                else:
                    s += num_str(nlinexpr.coeff, started) + '*'
            s += nlinexpr.fun_nlin.__name__ + '('
            for j, var in enumerate(nlinexpr.vars_fun):
                if var.expand:
                    s += '<'
                s += var.name
                if var.expand:
                    s += '>'
                if j != len(nlinexpr.vars_fun) - 1:
                    s += ','
            if nlinexpr.kwargs:
                s += ','
            for j, kwarg in enumerate(nlinexpr.kwargs.items()):
                s += kwarg[0] + '=' + str(kwarg[1])
                if j != len(nlinexpr.kwargs) - 1:
                    s += ','
            s += ')'
            if not started:
                started = True
        if self.const is not None and self.const != 0:
            if started and self.const > 0:
                s += ' + '
            s += num_str(self.const, started)
        if return_str is False:
            print(s)
        else:
            return s


# Variable type can be 'cont' for continuous (default), 'int' for integer, or 'bin' for binary
# full string names are also acceptable i.e. 'continuous', 'integer', and 'binary'
# bounds must be defined for variables involved in nonlinear expressions
class Var(Expr):
    def __init__(self, var_name, low_bound=None, up_bound=None, var_type='cont', expand=False,
                 breakpoint_fun=None, **breakpoint_kwargs):
        super().__init__()
        if isinstance(var_name, numbers.Number):
            self.const = var_name
            return
        check_string(var_name)
        check_boolean(expand, 'expand')
        check_breakpoint(breakpoint_fun, breakpoint_kwargs)
        var_type = check_vartype(var_type)
        coeff = 1.0
        bounds, expand = check_bounds(low_bound, up_bound, var_type, expand)
        self.singular = True
        self.name = var_name
        self.bounds = bounds
        self.vertices = None
        self.var_type = var_type
        self.expand = expand
        self.breakpoint_fun = None
        self.breakpoint_kwargs = breakpoint_kwargs
        self.var_info = VarInfo(coeff, bounds, var_type, var_name, expand, breakpoint_fun, breakpoint_kwargs)
        self.lin_vars[var_name] = self.var_info


class VarInfo:
    def __init__(self, coeff, bounds, var_type, name, expand, breakpoint_fun, breakpoint_kwargs):
        self.coeff = coeff
        self.bounds = bounds
        self.var_type = var_type
        self.name = name
        self.expand = expand
        self.decompose = False
        self.n_pieces = None
        self.n_vertices = None
        self.vertices = None
        self.decomposed_bins = None
        self.decomposed_vars = None
        self.value = None
        self.dvalue = None
        self.breakpoint_fun = breakpoint_fun
        self.breakpoint_kwargs = breakpoint_kwargs

    # n_pieces and n_vertices are not necessarily the ones set by the user if this is an expanded variable
    def set_vertices(self, vertices):
        check_vertices(vertices, self.bounds, self.var_type)
        self.vertices = vertices
        self.n_pieces = len(vertices) - 1
        self.n_vertices = len(vertices)
        self.decompose = True
        self.decomposed_vars = [VarInfo(1.0, [vertices[i], vertices[i+1]], self.var_type,
                                        self.name + '_var_' + str(i), self.expand, None, {})
                                for i in range(self.n_pieces)]
        self.decomposed_bins = [VarInfo(1.0, [None, None], BIN_VAR, self.name + '_bin_' + str(i), False, None, {})
                                for i in range(self.n_pieces)]

    def set_value(self, value):
        if self.value is not None:
            self.dvalue = value - self.value
        self.value = value


# vars_fun is an iterable returning Var objects that are involved in the nonlinear expression
# once initialized, vars_fun is a list of varinfo objects
# avoid passing objects (such as splines) as kwargs
class NlinExpr(Expr):
    def __init__(self, fun_nlin, *vars_fun, **kwargs):
        super().__init__()
        coeff = 1.0
        check_kwargs(kwargs)
        check_fun(fun_nlin)
        vars_fun = check_variables(vars_fun, Var)
        vars_fun = [var_fun.var_info for var_fun in vars_fun]
        vars_fun_names = tuple(var_fun.name for var_fun in vars_fun)
        self.nlinexprs[(fun_nlin, vars_fun_names, frozenset(kwargs.items()))] = \
            NlinExprInfo(coeff, fun_nlin, vars_fun, kwargs)


class NlinExprInfo:
    def __init__(self, coeff, fun_nlin, vars_fun, kwargs):
        self.coeff = coeff
        self.fun_nlin = fun_nlin
        self.vars_fun = vars_fun
        self.vars_fun_names = [var_fun.name for var_fun in vars_fun]

        self.kwargs = kwargs
        self.decomposed_vars = []
        self.decomposed_bins = []
        self.cube_ids = None
        self.simplex_ids = None
        self.nlinexpr_id = None
        self.sharing_nlinexpr_decomposition = None   # share nlinexpr decomposition with other nlinexpr or singular var

    def decompose_expression(self, decomposed_nlinexprs, variables_to_decompose):
        if self.nlinexpr_id is None:
            raise Exception('id of nonlinear expression is not yet set')

        self.vars_pieces = [var_fun.n_pieces for var_fun in self.vars_fun]
        self.cube_ids = list(itertools.product(*[range(vp) for vp in self.vars_pieces]))
        self.simplex_ids = list(itertools.permutations(range(len(self.vars_fun))))
        n_simplices = np.math.factorial(len(self.vars_fun))

        nlinexpr_vars_fun_names = [nlinexpr.vars_fun_names for nlinexpr in decomposed_nlinexprs]
        if self.vars_fun_names in nlinexpr_vars_fun_names:
            sharing_parent_index = nlinexpr_vars_fun_names.index(self.vars_fun_names)
        else:
            sharing_parent_index = None
        if len(self.vars_fun) == 1 or sharing_parent_index is not None:
            self.sharing_nlinexpr_decomposition = True
            if len(self.vars_fun) == 1:
                variable_to_decompose = variables_to_decompose[self.vars_fun[0].name]
                self.decomposed_vars = copy.deepcopy(np.array(variable_to_decompose.decomposed_vars))
                self.decomposed_vars = self.decomposed_vars[np.newaxis, :, np.newaxis]
                self.decomposed_bins = copy.deepcopy(np.array(variable_to_decompose.decomposed_bins))
                self.decomposed_bins = self.decomposed_bins[:, np.newaxis]
            else:
                self.decomposed_vars = copy.deepcopy(decomposed_nlinexprs[sharing_parent_index].decomposed_vars)
                self.decomposed_bins = copy.deepcopy(decomposed_nlinexprs[sharing_parent_index].decomposed_bins)
        else:
            self.sharing_nlinexpr_decomposition = False
            self.decomposed_vars = np.full([len(self.vars_fun)] + self.vars_pieces + [n_simplices], None)
            self.decomposed_bins = np.full(self.vars_pieces + [n_simplices], None)

        for cube_id in self.cube_ids:
            cube_id_str = ','.join([str(dimension_id) for dimension_id in cube_id])
            vertex_cube = [vf.vertices[cube_id[j]] for j, vf in enumerate(self.vars_fun)]
            f_vertex_cube = self.fun_nlin(*vertex_cube, **self.kwargs)
            for simplex_index_id, simplex_id in enumerate(self.simplex_ids):
                simplex_id_str = str(simplex_index_id)
                cum_const = 0
                vertex_a = copy.copy(vertex_cube)
                vertex_b = copy.copy(vertex_a)
                for dimension_step in simplex_id:
                    var_fun = self.vars_fun[dimension_step]
                    vertex_index = cube_id[dimension_step]
                    vertex_b[dimension_step] = var_fun.vertices[vertex_index + 1]
                    f_vertex_a = self.fun_nlin(*vertex_a, **self.kwargs)
                    f_vertex_b = self.fun_nlin(*vertex_b, **self.kwargs)
                    coeff = f_vertex_b - f_vertex_a
                    d = var_fun.vertices[vertex_index + 1] - var_fun.vertices[vertex_index]
                    coeff = coeff / d
                    cum_const += coeff * var_fun.vertices[vertex_index]
                    bounds = [var_fun.vertices[vertex_index], var_fun.vertices[vertex_index + 1]]
                    var_type = var_fun.var_type
                    expand = var_fun.expand
                    if self.sharing_nlinexpr_decomposition:
                        self.decomposed_vars[(dimension_step,) + cube_id + (simplex_index_id,)].coeff = coeff
                    else:
                        var_name = 'nlinexpr' + str(self.nlinexpr_id) + '_' + var_fun.name + \
                                   '_' + cube_id_str + '_' + simplex_id_str
                        var_info = VarInfo(coeff, bounds, var_type, var_name, expand, None, {})
                        self.decomposed_vars[(dimension_step,) + cube_id + (simplex_index_id,)] = var_info
                    vertex_a = copy.copy(vertex_b)
                if self.sharing_nlinexpr_decomposition:
                    self.decomposed_bins[cube_id + (simplex_index_id,)].coeff = f_vertex_cube - cum_const
                else:
                    bin_name = 'nlinexpr' + str(self.nlinexpr_id) + '_' + cube_id_str + '_' + simplex_id_str + '_bin'
                    bin_info = VarInfo(f_vertex_cube - cum_const, [None, None], BIN_VAR, bin_name, False, None, {})
                    self.decomposed_bins[cube_id + (simplex_index_id,)] = bin_info
        if not self.sharing_nlinexpr_decomposition:
            decomposed_nlinexprs.append(self)


def add_expr(obj, expr, return_copy=True):
    if return_copy:
        obj = copy.deepcopy(obj)
    obj.singular = False
    if not isinstance(expr, Var) and not isinstance(expr, NlinExpr) and not isinstance(expr, numbers.Number):
        raise TypeError('addition of unsupported expression ' + str(expr) + ' to object ' + str(obj))
    expr = copy.deepcopy(expr)
    if isinstance(expr, numbers.Number):
        if obj.const is None:
            obj.const = expr
        else:
            obj.const += expr
    else:
        combine_expr(obj, expr)
    return obj


def mul_expr(obj, coeff, return_copy=True):
    if isinstance(coeff, NlinExpr) or isinstance(coeff, Var):
        raise Exception('cannot multiply between variables/nonlinear expressions')
    check_coeff_number(coeff)
    if return_copy:
        obj = copy.deepcopy(obj)
    obj.singular = False
    multiply_coeffs_const(obj, coeff)
    return obj


def div_expr(obj, coeff, return_copy=True):
    if isinstance(coeff, NlinExpr) or isinstance(coeff, Var):
        raise ValueError('cannot multiply between variables/nonlinear expressions')
    check_coeff_number(coeff)
    if coeff == 0:
        raise ZeroDivisionError('dividing by 0')
    if return_copy:
        obj = copy.deepcopy(obj)
    obj.singular = False
    coeff = 1.0/coeff
    multiply_coeffs_const(obj, coeff)
    return obj


def sub_expr(obj, expr, return_copy=True):
    if not (isinstance(expr, Var) or isinstance(expr, NlinExpr) or isinstance(expr, numbers.Number)):
        raise ValueError('subtracting with unknown object')
    expr = copy.deepcopy(expr)
    if return_copy:
        obj = copy.deepcopy(obj)
    obj.singular = False
    if isinstance(expr, numbers.Number):
        if obj.const is None:
            obj.const = 0
        obj.const -= expr
    else:
        multiply_coeffs_const(expr, -1)
        combine_expr(obj, expr)
    return obj


def rsub_expr(obj, coeff):
    check_coeff_number(coeff)
    obj = copy.deepcopy(obj)
    obj.singular = False
    multiply_coeffs_const(obj, -1)
    if obj.const is None:
        obj.const = 0
    obj.const = coeff + obj.const
    return obj


def multiply_coeffs_const(obj, k):
    for var_name in obj.lin_vars:
        obj.lin_vars[var_name].coeff *= k
    for nlinexpr in obj.nlinexprs:
        obj.nlinexprs[nlinexpr].coeff *= k
    if obj.const is not None:
        obj.const *= k
    if k == 0:
        obj.lin_vars = {}
        obj.nlinexprs = {}
        obj.const = None


def combine_expr(obj, expr):
    for var_name in expr.lin_vars:
        if var_name in obj.lin_vars:
            check_var_consistency(obj.lin_vars[var_name], expr.lin_vars[var_name])
            obj.lin_vars[var_name].coeff += expr.lin_vars[var_name].coeff
        else:
            obj.lin_vars[var_name] = expr.lin_vars[var_name]
        if obj.lin_vars[var_name].coeff == 0:
            del(obj.lin_vars[var_name])
    # combine nonlinear expressions
    for key, nlinexpr_expr in expr.nlinexprs.items():
        # if nonlinear expressions with the same variables exist, then they must be consistent
        if key in obj.nlinexprs:
            nlinexpr_obj = obj.nlinexprs[key]
            for i in range(len(nlinexpr_obj.vars_fun)):
                check_var_consistency(nlinexpr_obj.vars_fun[i], nlinexpr_expr.vars_fun[i])
            nlinexpr_obj.coeff += nlinexpr_expr.coeff
        else:
            obj.nlinexprs[key] = nlinexpr_expr
        if obj.nlinexprs[key].coeff == 0:
            del(obj.nlinexprs[key])
    if obj.const is not None or expr.const is not None:
        obj.const = (obj.const if obj.const is not None else 0.0) + (expr.const if expr.const is not None else 0.0)

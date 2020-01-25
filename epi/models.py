""" Models. """

import numpy as np
import tensorflow as tf
from epi.error_formatters import format_type_err_msg

REAL_NUMERIC_TYPES = (int, float)

class Parameter:
    def __init__(self, name, bounds=(np.NINF, np.PINF)):
        self._set_name(name)
        self._set_bounds(bounds)

    def _set_name(self, name):
        if type(name) is not str:
            raise TypeError(format_type_err_msg(self, "name", name, str))
        self.name = name

    def _set_bounds(self, bounds):
        _type = type(bounds)
        if _type in [list, tuple]:
            len_bounds = len(bounds)
            if _type is list:
                bounds = tuple(bounds)
        elif _type is np.ndarray:
            len_bounds = bounds.shape[0]
            bounds = (bounds[0], bounds[1])
        else:
            raise TypeError(
                "Parameter argument bounds must be tuple, list, or numpy array not %s."
                % _type.__name__
            )

        if len_bounds != 2:
            raise ValueError("Parameter bounds arg must be length 2.")

        lb = bounds[0]
        ub = bounds[1]
        if (not isinstance(lb, REAL_NUMERIC_TYPES)):
            raise TypeError('Lower bound has type %s, not numeric.' % type(lb))
        if (not isinstance(ub, REAL_NUMERIC_TYPES)):
            raise TypeError('Upper bound has type %s, not numeric.' % type(ub))

        if (lb > ub):
            raise ValueError("Parameter %s lower bound is greater than upper bound." % self.name)
        elif (lb == ub):
            raise ValueError("Parameter %s lower bound is equal to upper bound." % self.name)

        self.bounds = bounds

"""
class Model:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        if not self.parameter_check(verbose=True):
            raise ValueError("Invalid parameter list.")
        self.eps = lambda: None

    def set_name(self, name):
        self.name = name

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_eps():
        raise NotImplementedError()

    def epi():
        raise NotImplementedError()

    def load_epi_dist():
        raise NotImplementedError()

    def parameter_check(self, verbose=False):
        d = dict()
        for param in self.parameters:
            name = param.name
            if name in d:
                if verbose:
                    print("Warning: Duplicate parameter %s in Model.parameters." % name)
                return False
            else:
                d[name] = True

            bounds = param.bounds
            if bounds[0] == bounds[1]:
                if verbose:
                    print(
                        "Warning: Left bound equal to right bound for parameter %s."
                        % name
                    )
                return False
            elif bounds[0] > bounds[1]:
                if verbose:
                    print(
                        "Warning: Left bound greater than right bound for parameter %s."
                        % name
                    )
                return False

        return True
"""

"""Convolution-based filter operators."""

import numpy as np
from dedalus.core.field import Operand
from dedalus.core.operators import Operator, FutureField


class Convolve(Operator, FutureField):
    """Basic convolution operator."""

    name = 'Conv'

    def meta_constant(self, axis):
        return (self.args[0].meta[axis]['constant'] and
                self.args[1].meta[axis]['constant'])

    def check_conditions(self):
        # Coefficient layout
        arg0, arg1 = self.args
        return ((arg0.layout == self._coeff_layout) and
                (arg1.layout == self._coeff_layout))

    def operate(self, out):
        arg0, arg1 = self.args
        arg0.require_coeff_space()
        arg1.require_coeff_space()
        # Multiply coefficients
        out.layout = self._coeff_layout
        np.multiply(arg0.data, arg1.data, out=out.data)


def build_filter(domain, method, parameter, N=np.inf, norm=2):
    """Build filter operator."""
    kn = domain.all_elements()
    kn = np.meshgrid(*[ki.ravel() for ki in kn], indexing='ij')
    knorm = np.linalg.norm(kn, axis=0, ord=norm)
    eta = domain.new_field(name='eta')
    if method=="sigma":
        eta['c'] = np.exp(-0.5*(knorm / parameter)**2)
    elif method=="epsilon":
        kcut = (N - 1) // 2
        eta['c'] = np.exp(np.log(parameter) * (knorm / kcut)**2)
    elif method=="mu":
        kcut = (N - 1) // 2
        eta['c'] = np.exp(-0.5 * (knorm / (parameter * kcut))**2)
    elif method=="sharp":
        eta['c'] = 1
        eta['c'][knorm > parameter] = 0
    else:
        raise ValueError("method must be either 'sigma', 'epsilon', 'mu', or 'sharp'")
    Filter = lambda field, eta=eta: Convolve(eta, field)
    return Filter
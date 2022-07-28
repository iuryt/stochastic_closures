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


def build_sharp_filter(domain, N, norm=2):
    """Build sharp filter operator."""
    kmax = (N - 1) // 2
    kn = domain.all_elements()
    kn = np.meshgrid(*[ki.ravel() for ki in kn], indexing='ij')
    knorm = np.linalg.norm(kn, axis=0, ord=norm)
    eta = domain.new_field(name='eta')
    eta['c'] = 1
    eta['c'][knorm > kmax] = 0
    Filter = lambda field, eta=eta: Convolve(eta, field)
    return Filter


def build_gaussian_filter(domain, N, epsilon, norm=2):
    """Build gaussian filter operator."""
    kcut = (N - 1) // 2
    kn = domain.all_elements()
    kn = np.meshgrid(*[ki.ravel() for ki in kn], indexing='ij')
    knorm = np.linalg.norm(kn, axis=0, ord=norm)
    eta = domain.new_field(name='eta')
    eta['c'] = np.exp(np.log(epsilon) * (knorm / kcut)**2)
    Filter = lambda field, eta=eta: Convolve(eta, field)
    return Filter



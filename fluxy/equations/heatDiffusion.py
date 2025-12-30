from fluxy.equations.baseEquation import BaseEquation
from fluxy.equations.numericalSchemes.firstOrderEuler import FirstOrderEuler
from fluxy.equations.numericalSchemes.secondOrderCentral import SecondOrderCentral

class HeatDiffusion(BaseEquation):
    def __init__(self, params, mesh, var_name):
        super().__init__(params, mesh, var_name)
        
        self.has_diffusion = True
        self.has_time_derivative = True

        self.dTdt = FirstOrderEuler(params, mesh, self.bc)
        self.d2Tdx2 = SecondOrderCentral(params, mesh, self.bc)

    def time_derivative(self, time, field):
        self.dTdt.apply(self.solver, time, field)

    def diffusion(self, time, field):
        alpha = self.params.solver('fluid', 'alpha')
        self.d2Tdx2.apply(self.solver, time, field, alpha)

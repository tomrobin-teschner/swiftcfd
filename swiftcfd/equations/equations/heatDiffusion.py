from swiftcfd.equations.baseEquation import BaseEquation
from swiftcfd.equations.numericalSchemes.implicit.firstOrderEuler import FirstOrderEuler
from swiftcfd.equations.numericalSchemes.implicit.secondOrderCentral import SecondOrderCentral

class HeatDiffusion(BaseEquation):
    def __init__(self, params, mesh):
        super().__init__(params, mesh, self.get_variable_name())
        
        self.has_first_order_time_derivative = True
        self.has_second_order_space_derivative = True

        self.dTdt = FirstOrderEuler(self.params, self.mesh, self.ic)
        self.d2Tdx2 = SecondOrderCentral(self.params, self.mesh, self.ic)

    def first_order_time_derivative(self, time, field):
        self.dTdt.apply(self.solver, time, field)

    def second_order_space_derivative(self, time, field):
        alpha = self.params('solver', 'fluid', 'alpha')
        self.d2Tdx2.apply(self.solver, time, field, alpha)
    
    def get_diffusion_coefficients(self):
        alpha = self.params('solver', 'fluid', 'alpha')
        return alpha

    def get_variable_name(self):
        return 'T'

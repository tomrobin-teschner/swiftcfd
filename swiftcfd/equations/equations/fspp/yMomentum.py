from swiftcfd.equations.equations.primitiveVariables import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.equations.numericalSchemes.numericalSchemesBase import WRT
from swiftcfd.equations.numericalSchemes.implicit.firstOrderEuler import FirstOrderEuler
from swiftcfd.equations.numericalSchemes.implicit.firstOrderUpwind import FirstOrderUpwind
from swiftcfd.equations.numericalSchemes.implicit.secondOrderCentral import SecondOrderCentral

class yMomentum(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_first_order_time_derivative = True
        self.has_first_order_space_derivative = True
        self.has_second_order_space_derivative = True

        self.requires_linearisation = True

        constructor_arguments = (self.params, self.mesh, self.ic, self.field_manager)

        self.dvdt = FirstOrderEuler(*constructor_arguments)
        self.dvdx = FirstOrderUpwind(*constructor_arguments)
        self.dvdy = FirstOrderUpwind(*constructor_arguments)

        self.d2vdx2 = SecondOrderCentral(*constructor_arguments)
        self.d2vdy2 = SecondOrderCentral(*constructor_arguments)

    def first_order_time_derivative(self, time):
        self.dvdt.apply(WRT.t, self.solver, time, self.get_variable_name())

    def first_order_space_derivative(self, time):
        self.dvdx.apply(WRT.x, self.solver, time, self.get_variable_name())
        self.dvdy.apply(WRT.y, self.solver, time, self.get_variable_name())

    def second_order_space_derivative(self, time):
        nu = self.params('solver', 'fluid', 'nu')
        self.d2vdx2.apply(WRT.x, self.solver, time, self.get_variable_name(), -1.0 * nu)
        self.d2vdy2.apply(WRT.y, self.solver, time, self.get_variable_name(), -1.0 * nu)
    
    def get_diffusion_coefficients(self):
        nu = self.params('solver', 'fluid', 'nu')
        return nu

    def get_variable_name(self):
        return pv.velocity_y.name()

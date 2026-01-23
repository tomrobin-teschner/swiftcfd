from swiftcfd.equations.equations.primitiveVariables import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.equations.numericalSchemes.numericalSchemesBase import WRT
from swiftcfd.equations.numericalSchemes.implicit.firstOrderEuler import FirstOrderEuler
from swiftcfd.equations.numericalSchemes.implicit.secondOrderBackwards import SecondOrderBackwards
from swiftcfd.equations.numericalSchemes.implicit.firstOrderUpwind import FirstOrderUpwind
from swiftcfd.equations.numericalSchemes.implicit.secondOrderUpwind import SecondOrderUpwind
from swiftcfd.equations.numericalSchemes.implicit.secondOrderCentral import SecondOrderCentral

class xMomentum(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_first_order_time_derivative = True
        self.has_first_order_space_derivative = True
        self.has_second_order_space_derivative = True

        self.requires_linearisation = True

        constructor_arguments = (self.params, self.mesh, self.ic, self.field_manager)

        self.dudt = SecondOrderBackwards(*constructor_arguments)
        self.dudx = SecondOrderUpwind(*constructor_arguments)
        self.dudy = SecondOrderUpwind(*constructor_arguments)

        self.d2udx2 = SecondOrderCentral(*constructor_arguments)
        self.d2udy2 = SecondOrderCentral(*constructor_arguments)

    def first_order_time_derivative(self, time):
        self.dudt.apply(WRT.t, self.solver, time, self.get_variable_name())

    def first_order_space_derivative(self, time):
        self.dudx.apply(WRT.x, self.solver, time, self.get_variable_name())
        self.dudy.apply(WRT.y, self.solver, time, self.get_variable_name())

    def second_order_space_derivative(self, time):
        nu = self.params('solver', 'fluid', 'nu')
        self.d2udx2.apply(WRT.x, self.solver, time, self.get_variable_name(), -1.0 * nu)
        self.d2udy2.apply(WRT.y, self.solver, time, self.get_variable_name(), -1.0 * nu)
    
    def get_diffusion_coefficients(self):
        nu = self.params('solver', 'fluid', 'nu')
        return nu

    def get_variable_name(self):
        return pv.velocity_x.name()

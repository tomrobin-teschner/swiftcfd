from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.enums import WRT
from swiftcfd.equations.numericalSchemes.numericalSchemeFactory import NumericalSchemeFactory
from swiftcfd.gradients.firstOrderGradient import FirstOrderGradient as gradient

class Pressure(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_second_order_space_derivative = True
        self.has_source = True

        numerical_schemes = NumericalSchemeFactory(self.params, self.mesh, self.bc, self.field_manager)

        self.d2pdx2 = numerical_schemes.create_second_order_space_derivative_scheme(self)
        self.d2pdy2 = numerical_schemes.create_second_order_space_derivative_scheme(self)

        self.grad_u = gradient(self.mesh, self.field_manager, pv.velocity_x.name())
        self.grad_v = gradient(self.mesh, self.field_manager, pv.velocity_y.name())
        self.grad_p = gradient(self.mesh, self.field_manager, pv.pressure.name())

        self.uses_second_order_time_integration = \
            self.params('solver', 'schemes', 'timeIntegrationScheme') == 'secondOrderBackwards'

    def second_order_space_derivative(self, runtime):
        self.d2pdx2.apply(WRT.x, self.solver, runtime, self.get_variable_name())
        self.d2pdy2.apply(WRT.y, self.solver, runtime, self.get_variable_name())

    def source(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt

        self.grad_u.compute()
        self.grad_v.compute()

        for block in range(0, self.mesh.num_blocks):
            for (i, j) in self.mesh.loop_cells(block):
                index = self.mesh.map3Dto1D(block, i, j)
                divergence = self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j]

                if self.uses_second_order_time_integration and runtime.current_timestep > 1:
                    rhs = (3.0 * rho / (2.0 * dt)) * divergence
                else:
                    rhs = (rho / dt) * divergence
                
                self.solver.add_to_b(index, rhs)
            
    def post_solve_task(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt
        self.grad_p.compute()

        if self.uses_second_order_time_integration and runtime.current_timestep > 1:
            multiplier = (2.0 * dt) / (3.0 * rho)
        else:
            multiplier = dt / rho

        for (block, i, j) in self.mesh.loop_all_cells():
            u = self.field_manager.fields[pv.velocity_x.name()][block, i, j] - multiplier * self.grad_p.x[block, i, j]
            v = self.field_manager.fields[pv.velocity_y.name()][block, i, j] - multiplier * self.grad_p.y[block, i, j]

            self.field_manager.fields[pv.velocity_x.name()][block, i, j] = u
            self.field_manager.fields[pv.velocity_y.name()][block, i, j] = v

    def get_diffusion_coefficients(self):
        pass

    def get_variable_name(self):
        return pv.pressure.name()

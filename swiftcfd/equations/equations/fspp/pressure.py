from swiftcfd.equations.equations.primitiveVariables import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.equations.numericalSchemes.numericalSchemesBase import WRT
from swiftcfd.equations.numericalSchemes.numericalSchemeFactory import NumericalSchemeFactory
from swiftcfd.gradients.firstOrderGradient import FirstOrderGradient as gradient

class Pressure(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_second_order_space_derivative = True
        self.has_source = True

        numerical_schemes = NumericalSchemeFactory(self.params, self.mesh, self.ic, self.field_manager)

        self.d2pdx2 = numerical_schemes.create_second_order_space_derivative_scheme(self)
        self.d2pdy2 = numerical_schemes.create_second_order_space_derivative_scheme(self)

        self.grad_u = gradient(self.mesh, self.field_manager, pv.velocity_x.name())
        self.grad_v = gradient(self.mesh, self.field_manager, pv.velocity_y.name())
        self.grad_p = gradient(self.mesh, self.field_manager, pv.pressure.name())

    def second_order_space_derivative(self, time):
        self.d2pdx2.apply(WRT.x, self.solver, time, self.get_variable_name())
        self.d2pdy2.apply(WRT.y, self.solver, time, self.get_variable_name())

    def source(self, time):
        rho = self.params('solver', 'fluid', 'rho')
        dt = time.dt

        self.grad_u.compute()
        self.grad_v.compute()

        for (block, i, j) in self.mesh.internal_loop_all_blocks():
            index = self.mesh.map3Dto1D(block, i, j)
            # uip1 = self.field_manager.fields[pv.velocity_x.name()][block, i + 1, j]
            # ui   = self.field_manager.fields[pv.velocity_x.name()][block, i, j]
            # uim1 = self.field_manager.fields[pv.velocity_x.name()][block, i - 1, j]
            
            # vjp1 = self.field_manager.fields[pv.velocity_y.name()][block, i, j + 1]
            # vj   = self.field_manager.fields[pv.velocity_y.name()][block, i, j]
            # vjm1 = self.field_manager.fields[pv.velocity_y.name()][block, i, j - 1]

            # pip1 = self.field_manager.fields[pv.pressure.name()][block, i + 1, j]
            # pi   = self.field_manager.fields[pv.pressure.name()][block, i, j]
            # pim1 = self.field_manager.fields[pv.pressure.name()][block, i - 1, j]

            # pjp1 = self.field_manager.fields[pv.pressure.name()][block, i, j + 1]
            # pj   = self.field_manager.fields[pv.pressure.name()][block, i, j]
            # pjm1 = self.field_manager.fields[pv.pressure.name()][block, i, j - 1]

            # dx, dy = self.mesh.get_spacing(block)

            # p_x_east = (pip1 - pi) / dx
            # p_x_west = (pi - pim1) / dx

            # p_y_north = (pjp1 - pj) / dy
            # p_y_south = (pj - pjm1) / dy

            # ue = 0.5 * (uip1 + ui) - (dt / rho) * p_x_east
            # uw = 0.5 * (ui + uim1) - (dt / rho) * p_x_west

            # vn = 0.5 * (vjp1 + vj) - (dt / rho) * p_y_north
            # vs = 0.5 * (vj + vjm1) - (dt / rho) * p_y_south

            # grad_u_x = (ue - uw) / dx
            # grad_v_y = (vn - vs) / dy

            # rhs = (rho / dt) * (grad_u_x + grad_v_y)

            # print(equations[0].solver.A.getDiagonal().getArray())
            # print(equations[0].solver.A.getDiagonal().getArray().min())
            # print(equations[0].solver.A.getDiagonal().getArray().max())
            # print(len(equations[0].solver.A.getDiagonal().getArray()))

                
            
            # currently missing rhie-chow interpolation
            rhs = (rho / dt) * (self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j])
            self.solver.add_to_b(index, rhs)
            
    def post_solve_task(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt
        self.grad_p.compute()

        for (block, i, j) in self.mesh.internal_loop_all_blocks():
            u = self.field_manager.fields[pv.velocity_x.name()][block, i, j] - (dt / rho) * self.grad_p.x[block, i, j]
            v = self.field_manager.fields[pv.velocity_y.name()][block, i, j] - (dt / rho) * self.grad_p.y[block, i, j]

            self.field_manager.fields[pv.velocity_x.name()][block, i, j] = u
            self.field_manager.fields[pv.velocity_y.name()][block, i, j] = v

    def get_diffusion_coefficients(self):
        pass

    def get_variable_name(self):
        return pv.pressure.name()

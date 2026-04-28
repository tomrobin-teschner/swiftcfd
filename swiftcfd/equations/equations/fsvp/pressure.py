from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.enums import WRT
from swiftcfd.equations.numericalSchemes.numericalSchemeFactory import NumericalSchemeFactory
from swiftcfd.gradients.firstOrderGradient import FirstOrderGradient as gradient


class Pressure(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_first_order_time_derivative = True
        self.has_second_order_space_derivative = True
        self.has_source = True

        numerical_schemes = NumericalSchemeFactory(self.params, self.mesh, self.bc, self.field_manager)

        self.dpdt = numerical_schemes.create_time_integration_scheme(self)
        self.d2pdx2 = numerical_schemes.create_second_order_space_derivative_scheme_explicit(self)
        self.d2pdy2 = numerical_schemes.create_second_order_space_derivative_scheme_explicit(self)

        self.grad_u = gradient(self.mesh, self.field_manager, pv.velocity_x.name())
        self.grad_v = gradient(self.mesh, self.field_manager, pv.velocity_y.name())
        self.grad_p = gradient(self.mesh, self.field_manager, pv.pressure.name())

        self.uses_second_order_time_integration = \
            self.params('solver', 'schemes', 'timeIntegrationScheme') == 'secondOrderBackwards'
        
        f = open("divergence.txt", "w")
        f.close()

        
    # def first_order_time_derivative(self, runtime):
    #     self.dpdt.apply(WRT.t, self.solver, runtime, self.get_variable_name())
    
    def first_order_time_derivative(self, runtime):
        dt = runtime.dt
        for block in range(0, self.mesh.num_blocks):
            for (i, j) in self.mesh.loop_cells(block):
                index = self.mesh.map3Dto1D(block, i, j)
                self.solver.add_to_A(index, index, 1.0 / dt)
                self.solver.add_to_b(index, (1.0 / dt) * self.field_manager.fields[pv.pressure.name()][block, i, j])

    def second_order_space_derivative(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt

        self.d2pdx2.apply(WRT.x, self.solver, runtime, self.get_variable_name(), dt / rho)
        self.d2pdy2.apply(WRT.y, self.solver, runtime, self.get_variable_name(), dt / rho)

    def source(self, runtime):
        self.grad_u.compute()
        self.grad_v.compute()

        for block in range(0, self.mesh.num_blocks):
            for (i, j) in self.mesh.loop_cells(block):
                index = self.mesh.map3Dto1D(block, i, j)
                divergence = self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j]                
                self.solver.add_to_b(index, -1.0 * divergence)
        



        total_div = 0.0
        for block in range(0, self.mesh.num_blocks):
            for (i, j) in self.mesh.loop_cells(block):
                index = self.mesh.map3Dto1D(block, i, j)
                divergence = self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j]
                total_div += abs(divergence)
                self.solver.add_to_b(index, -1.0 * divergence)
        with open("divergence.txt", "a") as f:
            f.write(f"  mean|div u*| = {total_div / self.mesh.total_cells:.6e}\n")
    
    def solve(self, runtime):
        n_pseudo = 1
        for _ in range(n_pseudo):
            self.solver.reset_A()
            self.solver.reset_b()
            
            if self.has_first_order_time_derivative:
                self.first_order_time_derivative(runtime)
            if self.has_second_order_space_derivative:
                self.second_order_space_derivative(runtime)
            if self.has_source:
                self.source(runtime)
            
            self.solver.assemble()
            self.solver.solve(self.field_manager.fields[self.get_variable_name()])

    def post_solve_task(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt

        self.grad_p.compute()





        # ADD THIS
        u_max_before = max(abs(self.field_manager.fields[pv.velocity_x.name()][block, i, j])
                       for (block, i, j) in self.mesh.loop_all_cells())






        if self.uses_second_order_time_integration and runtime.current_timestep > 1:
            multiplier = (2.0 * dt) / (3.0 * rho)
        else:
            multiplier = dt / rho

        for (block, i, j) in self.mesh.loop_all_cells():
            u = self.field_manager.fields[pv.velocity_x.name()][block, i, j] - multiplier * self.grad_p.x[block, i, j]
            v = self.field_manager.fields[pv.velocity_y.name()][block, i, j] - multiplier * self.grad_p.y[block, i, j]

            self.field_manager.fields[pv.velocity_x.name()][block, i, j] = u
            self.field_manager.fields[pv.velocity_y.name()][block, i, j] = v
        





        # ADD THIS
        u_max_after = max(abs(self.field_manager.fields[pv.velocity_x.name()][block, i, j])
                        for (block, i, j) in self.mesh.loop_all_cells())
        dp_max = max(abs(self.grad_p.x[block, i, j])
                    for (block, i, j) in self.mesh.loop_all_cells())

        total_div_after = 0.0
        self.grad_u.compute()  # recompute on corrected velocity
        self.grad_v.compute()
        for (block, i, j) in self.mesh.loop_all_cells():
            total_div_after += abs(self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j])
        
        p_mean = sum(self.field_manager.fields[pv.pressure.name()][block, i, j] 
                    for (block, i, j) in self.mesh.loop_all_cells()) / self.mesh.total_cells
        p_max = max(abs(self.field_manager.fields[pv.pressure.name()][block, i, j])
                    for (block, i, j) in self.mesh.loop_all_cells())
        
        with open("divergence.txt", "a") as f:
            f.write(f"  mean|div u_corrected| = {total_div_after / self.mesh.total_cells:.6e}")
            f.write(f"  p_mean = {p_mean:.6e}  p_max = {p_max:.6e}\n")
            f.write(f"  u_max before={u_max_before:.4e}  after={u_max_after:.4e}")
            f.write(f"  dp_max={dp_max:.4e}  multiplier={multiplier:.4e}\n")

    def get_diffusion_coefficients(self):
        pass

    def get_variable_name(self):
        return pv.pressure.name()

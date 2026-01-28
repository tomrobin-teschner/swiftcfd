from sys import float_info
from math import pow
from swiftcfd.enums import BCType
from swiftcfd.enums import PrimitiveVariables as pv

class Runtime():
    def __init__(self, params, mesh, field_manager, equations):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager
        self.equations = equations

        # check if velocity is a primitive variable
        self.has_advection = any(eq.get_variable_name() == pv.velocity_x.name() or eq.get_variable_name() == pv.velocity_y.name() for eq in self.equations)

        # check if viscous dissipation is present
        self.has_diffusion = any(eq.has_second_order_space_derivative for eq in self.equations)

        # linearisation related settings
        self.num_picard_iterations = self.params('solver', 'convergence', 'picardIterations')
        self.current_picard_iteration = 0
        
        # time information
        self.total_timesteps = self.params('solver', 'time', 'timesteps')
        
        self.current_time = 0.0
        self.current_timestep = 0

        self.dt = self.params('solver', 'time', 'dt')
        self.CFL = 0.0

        # set up class
        self.__max_dirichlet_bc_value()
        if self.has_diffusion:
            self.__read_diffusion_coefficient()

    def __max_dirichlet_bc_value(self):
        # get maximum dirichlet value from boundary conditions
        self.max_dirichlet_value = 0
        for eqn in self.equations:
            for index, BCs in enumerate(eqn.bc.bc_type):
                for key, value in BCs.items():
                    if value == BCType.dirichlet:
                        dirichlet_value = eqn.bc.bc_value[index][key]
                        if dirichlet_value > self.max_dirichlet_value:
                            self.max_dirichlet_value = dirichlet_value
    
    def __read_diffusion_coefficient(self):
        self.diffusion_coefficient = 0.0

        # check for heat diffusion
        try:
            alpha = self.params('solver', 'fluid', 'alpha')
            self.diffusion_coefficient = alpha
        except:
            alpha = 0.0

        # check for viscous dissipation
        try:
            nu = self.params('solver', 'fluid', 'nu')
            self.diffusion_coefficient = nu
        except:
            nu = 0.0

    def compute_CFL(self):
        CFL_diffusion = 0.0
        CFL_advection = 0.0
        
        # now compute new time step value
        if self.has_diffusion:
            gamma = self.diffusion_coefficient
            CFL_diffusion = self.dt * gamma / pow(self.mesh.get_min_spacing(), 2)

        if self.has_advection:
            CFL_advection = float_info.max
            for block in range(0, self.mesh.num_blocks):
                dx, dy = self.mesh.get_spacing(block)
                for (i, j) in self.mesh.internal_loop_single_block(block):
                    u_velocity = self.field_manager.get_field(pv.velocity_x.name())[block, i, j]
                    v_velocity = self.field_manager.get_field(pv.velocity_y.name())[block, i, j]
                    u_mag = pow(u_velocity, 2) + pow(v_velocity, 2)

                    temp_CFL = u_mag * self.dt / min(dx, dy)
                    if temp_CFL < CFL_advection:
                        CFL_advection = temp_CFL
            CFL_at_BC = self.max_dirichlet_value * self.dt / min(dx, dy)
            CFL_advection = min(CFL_advection, CFL_at_BC)

        self.CFL = max(CFL_diffusion, CFL_advection)

    def has_not_reached_final_time(self):
        if self.current_timestep == self.total_timesteps:
            return False
        else:
            return True

    def has_not_reached_final_picard_iteration(self):
        self.current_picard_iteration += 1
        if self.current_picard_iteration > self.num_picard_iterations:
            self.current_picard_iteration = 0
            return False
        else:
            return True

    def is_final_picard_iteration(self):
        return self.current_picard_iteration == self.num_picard_iterations

    def update_time(self):
        self.current_time += self.dt
        self.current_timestep += 1

    
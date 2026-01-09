from sys import float_info
from math import pow
from swiftcfd.equations.boundaryConditions.boundaryConditions import BCType
from swiftcfd.equations.equations.primitiveVariables import PrimitiveVariables as pv

class Time():
    def __init__(self, params, mesh, field_manager, equations):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager
        self.equations = equations

        # check if velocity is a primitive variable
        self.has_advection = any(eq.var_name == pv.velocity_x.name() or eq.var_name == pv.velocity_y.name() for eq in self.equations)

        # check if viscous dissipation is present
        self.has_diffusion = any(eq.has_second_order_space_derivative for eq in self.equations)

        self.cfl_based_timestepping = self.params('solver', 'time', 'CFLBasedTimeStepping')
        self.end_time = self.params('solver', 'time', 'endTime')

        self.current_time = 0.0
        self.timestep = 0

        self.CFL = 0.0
        self.dt = 0.0

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

    def compute_dt(self):
        if self.has_diffusion:
            gamma = self.diffusion_coefficient
            if self.cfl_based_timestepping:
                CFL_diffusion = self.params('solver', 'time', 'CFL')
                dt_diffusion = CFL_diffusion * pow(self.mesh.get_min_spacing(), 2) / gamma
            else:
                dt_diffusion = self.params('solver', 'time', 'dt')
                CFL_diffusion = dt_diffusion * gamma / pow(self.mesh.get_min_spacing(), 2)

        if self.has_advection:
            if self.cfl_based_timestepping:
                CFL_advection = self.params('solver', 'time', 'CFL')
                
                dt_advection = float_info.max
                for block in range(0, self.mesh.num_blocks):
                    dx, dy = self.mesh.get_spacing(block)
                    for (i, j) in self.mesh.internal_loop_single_block(block):
                        u_velocity = self.field_manager.get_field(pv.velocity_x.name())[block, i, j]
                        v_velocity = self.field_manager.get_field(pv.velocity_y.name())[block, i, j]
                        u_mag = pow(u_velocity, 2) + pow(v_velocity, 2)

                        temp_dt = CFL_advection * min(dx, dy) / u_mag
                        if temp_dt < dt_advection:
                            dt_advection = temp_dt
                dt_at_BC = CFL_advection * min(dx, dy) / self.max_dirichlet_value
                dt_advection = min(dt_advection, dt_at_BC)
            else:
                dt_advection = self.params('solver', 'time', 'dt')

                CFL_advection = float_info.max
                for block in range(0, self.mesh.num_blocks):
                    dx, dy = self.mesh.get_spacing(block)
                    for (i, j) in self.mesh.internal_loop_single_block(block):
                        u_velocity = self.field_manager.get_field(pv.velocity_x.name())[block, i, j]
                        v_velocity = self.field_manager.get_field(pv.velocity_y.name())[block, i, j]
                        u_mag = pow(u_velocity, 2) + pow(v_velocity, 2)

                        temp_CFL = u_mag * dt_advection / min(dx, dy)
                        if temp_CFL < CFL_advection:
                            CFL_advection = temp_CFL
                CFL_at_BC = self.max_dirichlet_value * dt_advection / min(dx, dy)
                CFL_advection = min(CFL_advection, CFL_at_BC)
        
        if self.has_advection == True and self.has_diffusion == False:
            self.dt = dt_advection
            self.CFL = CFL_advection
        elif self.has_advection == False and self.has_diffusion == True:
            self.dt = dt_diffusion
            self.CFL = CFL_diffusion
        else:
            self.dt = min(dt_diffusion, dt_advection)
            self.CFL = min(CFL_diffusion, CFL_advection)

    def not_reached_end_time(self):
        return self.current_time < self.end_time

    def update_time(self):
        self.current_time += self.dt
        self.timestep += 1

    
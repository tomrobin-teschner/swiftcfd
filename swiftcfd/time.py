from sys import float_info
from math import pow

class Time():
    def __init__(self, params, mesh, eqn):
        self.params = params
        self.mesh = mesh
        self.has_advection = eqn.has_first_order_space_derivative
        self.has_diffusion = eqn.has_second_order_space_derivative

        if self.has_diffusion:
            self.diffusion_coefficient = eqn.get_diffusion_coefficients()

        self.cfl_based_timestepping = self.params('solver', 'time', 'CFLBasedTimeStepping')
        self.end_time = self.params('solver', 'time', 'endTime')

        self.current_time = 0.0
        self.timestep = 0

        self.CFL = 0.0
        self.dt = 0.0   
    
    def compute_dt(self, *fields):
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
                for field in fields:
                    for block in range(0, self.mesh.num_blocks):
                        dx, dy = self.mesh.get_spacing(block_id)
                        for (block_id, i, j) in self.mesh.internal_loop_all_blocks(block):
                            temp_dt = CFL_advection * min(dx, dy) / field[block_id][i][j]
                            if temp_dt < dt_advection:
                                dt_advection = temp_dt
            else:
                dt_advection = self.params('solver', 'time', 'dt')

                CFL_advection = float_info.max
                for field in fields:
                    for block in range(0, self.mesh.num_blocks):
                        dx, dy = self.mesh.get_spacing(block_id)
                        for (block_id, i, j) in self.mesh.internal_loop_all_blocks(block):
                            temp_CFL = field[block_id][i][j] * dt_advection / min(dx, dy)
                            if temp_CFL < CFL_advection:
                                CFL_advection = temp_CFL
        
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

    
from sys import float_info
from math import pow

class Time():
    def __init__(self, params, mesh, has_advection, has_diffusion):
        self.params = params
        self.mesh = mesh
        self.has_advection = has_advection
        self.has_diffusion = has_diffusion

        self.cfl_based_timestepping = self.params.solver('time', 'CFLBasedTimeStepping')
        self.end_time = self.params.solver('time', 'endTime')
        self.current_time = 0.0        
    
    def compute_dt(self, *fields):
        if self.has_diffusion:
            nu = self.params.solver('fluid', 'nu')
            if self.cfl_based_timestepping:
                CFL_diffusion = self.params.solver('time', 'CFL')
                dt_diffusion = CFL_diffusion * pow(self.mesh.get_min_spacing(), 2) / nu
            else:
                dt_diffusion = self.params.solver('time', 'dt')
                CFL_diffusion = dt_diffusion * nu / pow(self.mesh.get_min_spacing(), 2)
        
        if self.has_advection:
            if self.cfl_based_timestepping:
                CFL_advection = self.params.solver('time', 'CFL')
                
                dt_advection = float_info.max
                for field in fields:
                    for block in range(0, self.mesh.num_blocks):
                        dx, dy = self.mesh.get_spacing(block_id)
                        for (block_id, i, j) in self.mesh.internal_loop_all_blocks(block):
                            temp_dt = CFL_advection * min(dx, dy) / field[block_id][i][j]
                            if temp_dt < dt_advection:
                                dt_advection = temp_dt
            else:
                dt_advection = self.params.solver('time', 'dt')

                CFL_advection = float_info.max
                for field in fields:
                    for block in range(0, self.mesh.num_blocks):
                        dx, dy = self.mesh.get_spacing(block_id)
                        for (block_id, i, j) in self.mesh.internal_loop_all_blocks(block):
                            temp_CFL = field[block_id][i][j] * dt_advection / min(dx, dy)
                            if temp_CFL < CFL_advection:
                                CFL_advection = temp_CFL
        
        if self.has_advection == True and self.has_diffusion == False:
            return df_advection, CFL_advection
        elif self.has_advection == False and self.has_diffusion == True:
            return dt_diffusion, CFL_diffusion
        else:
            return min(dt_diffusion, dt_advection), min(CFL_diffusion, CFL_advection)

    def not_reached_end_time(self):
        return self.current_time < self.end_time
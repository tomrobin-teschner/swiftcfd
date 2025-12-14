from math import pow

class Time():
    def __init__(self, params, mesh):
        self.params = params
        self.mesh = mesh
        self.addaptive_time_stepping = self.params.solver('time', 'adaptiveTimeStepping')
        self.end_time = self.params.solver('time', 'endTime')
        self.current_time = 0.0        
    
    def compute_dt(self):
        nu = self.params.solver('fluid', 'nu')
        if self.addaptive_time_stepping:
            CFL = self.params.solver('time', 'CFL')
            dt = CFL * pow(self.mesh.get_min_spacing(), 2) / nu
        else:
            dt = self.params.solver('time', 'dt')
            CFL = dt * nu / pow(self.mesh.get_min_spacing(), 2)
        return dt, CFL

    def reached_end_time(self):
        return self.current_time >= self.end_time
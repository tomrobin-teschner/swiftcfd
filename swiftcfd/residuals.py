from os.path import join

import numpy as np
import pandas as pd

class Residuals:
    def __init__(self, params, field_manager):
        self.params = params
        self.field_manager = field_manager
        
        self.picard_tolerance = {}
        self.picard_norm = {}
        self.picard_current_residual = {}
        self.picard_final_residual = {}
        self.num_picard_iterations = self.params('solver', 'convergence', 'picardIterations')

        self.convergence_tolerance = {}
        self.convergence_norm = {}
        self.convergence_residual = {}

        for key, value in self.params('solver', 'convergence', 'picard_tolerance').items():
            self.picard_tolerance[key] = value
            self.picard_norm[key] = 0
            self.picard_current_residual[key] = 0
            self.picard_final_residual[key] = []

        for key, value in self.params('solver', 'convergence', 'convergence_tolerance').items():
            self.convergence_tolerance[key] = value
            self.convergence_norm[key] = 0
            self.convergence_residual[key] = []
    
    def check_picard_convergence(self, runtime):
        for var_name, field in self.field_manager.get_all_fields().items():
            self.picard_current_residual[var_name] = np.linalg.norm(field._data - field.picard_old._data)
        
        if runtime.current_picard_iteration == 1:
            for var_name, residual in self.picard_current_residual.items():
                if residual > 0.0:
                    self.picard_norm[var_name] = residual
                else:
                    self.picard_norm[var_name] = 1.0
        
        for var_name in self.picard_current_residual:
            self.picard_current_residual[var_name] /= self.picard_norm[var_name]
        
        if runtime.current_picard_iteration == self.num_picard_iterations:
            for var_name, residual in self.picard_current_residual.items():
                self.picard_final_residual[var_name].append(residual)
        
        has_converged = True
        for var_name in self.picard_current_residual:
            if self.picard_current_residual[var_name] > self.picard_tolerance[var_name]:
                has_converged = False

        return has_converged

    def check_convergence(self, runtime):
        for var_name, field in self.field_manager.get_all_fields().items():
            self.convergence_residual[var_name].append(np.linalg.norm(field._data - field.old._data))
        
        if runtime.current_timestep == 1:
            for var_name, value in self.convergence_residual.items():
                if value[0] > 0.0:
                    self.convergence_norm[var_name] = value[0]
                else:
                    self.convergence_norm[var_name] = 1.0
        
        for var_name in self.convergence_residual:
            self.convergence_residual[var_name][-1] /= self.convergence_norm[var_name]
        
        has_converged = True
        for var_name in self.convergence_residual:
            if self.convergence_residual[var_name][-1] > self.convergence_tolerance[var_name]:
                has_converged = False

        return has_converged
    
    def write(self):
        headers = []
        for var_name in self.picard_current_residual:
            headers.append(f'{var_name}')
        df = pd.DataFrame(columns=headers)

        for var_name in self.picard_current_residual:
            df[f'{var_name}'] = self.convergence_residual[var_name]

        case = self.params('solver', 'output', 'filename')
        out_folder = join('output', case)
        df.to_csv(join(out_folder, 'residuals.csv'), index=False)

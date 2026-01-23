from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase

class SecondOrderBackwards(NumericalSchemesBase):
    def __init__(self, params, mesh, ic, field_manager):
        super().__init__(params, mesh, ic, field_manager)

    def _compute_coefficients(self, direction, runtime, var_name, multiplier):
        dt = runtime.dt
        dt_old = runtime.old_dt
        if runtime.timestep > 0:
            for block_id in range(0, self.mesh.num_blocks):
                self.coefficients.append({
                    'ap': 1.0/(dt + dt_old) * multiplier,
                    'b':  1.0/(dt + dt_old) * multiplier,
                    'is_second_order': True
                })
        else:
            for block_id in range(0, self.mesh.num_blocks):
                self.coefficients.append({
                    'ap': 1.0/dt * multiplier,
                    'b':  1.0/dt * multiplier,
                    'is_second_order': False
                })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            b_index  = self.mesh.map3Dto1D(block_id, i, j)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_b(b_index, self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j])
    
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        block1, i1, j1 = ij
        multiplier = self.coefficients[block1]['b']
        is_second_order = self.coefficients[block1]['is_second_order']

        if is_second_order:
            return multiplier * (4.0 * self.field_manager.fields[var_name].old[block1, i1, j1] - \
                self.field_manager.fields[var_name].oldold[block1, i1, j1])
        else:
            return multiplier * self.field_manager.fields[var_name].old[block1, i1, j1]

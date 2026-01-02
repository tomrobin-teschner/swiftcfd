from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase

class FirstOrderEuler(NumericalSchemesBase):
    def __init__(self, params, mesh, bc):
        super().__init__(params, mesh, bc)

    def _compute_coefficients(self, time, field, multiplier):
        dt, CFL = time.compute_dt(field)
        for block_id in range(0, self.mesh.num_blocks):
            self.coefficients.append({
                'ap': 1.0/dt * multiplier,
                'b':  1.0/dt * multiplier
            })
    
    def _compute_interior(self, block_id, solver, field):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            b_index  = self.mesh.map3Dto1D(block_id, i, j)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_b(b_index, self.coefficients[block_id]['b'] * field.old[block_id, i, j])
    
    def get_right_hand_side_contribution(self, ij, ip1j, im1j, ijp1, ijm1, field):
        block1, i1, j1 = ij
        multiplier = self.coefficients[block1]['b']
        return multiplier * field.old[block1, i1, j1]

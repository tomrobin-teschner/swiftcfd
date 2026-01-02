from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc):
        super().__init__(params, mesh, bc)

    def _compute_coefficients(self, time, field, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'b': multiplier
            })
    
    def _compute_interior(self, block_id, solver, field):
        dx, dy = self.mesh.get_spacing(block_id)
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            b_index = self.mesh.map3Dto1D(block_id, i, j)
            
            Tij  = field.old[block_id, i, j]
            Tip1 = field.old[block_id, i + 1, j]
            Tim1 = field.old[block_id, i - 1, j]
            Tjp1 = field.old[block_id, i, j + 1]
            Tjm1 = field.old[block_id, i, j - 1]
            alpha = self.coefficients[block_id]['b']

            grad_x = alpha * (Tip1 - 2.0 * Tij + Tim1) / pow(dx, 2)
            grad_y = alpha * (Tjp1 - 2.0 * Tij + Tjm1) / pow(dy, 2)

            solver.add_to_b(b_index, grad_x + grad_y)
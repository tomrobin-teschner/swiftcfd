from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc):
        super().__init__(params, mesh, bc)

    def _compute_coefficients(self, time, field, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'ap': multiplier * ( 2.0/pow(dx, 2) + 2.0/pow(dy, 2)),
                'ae': multiplier * (-1.0/pow(dx, 2)),
                'aw': multiplier * (-1.0/pow(dx, 2)),
                'an': multiplier * (-1.0/pow(dy, 2)),
                'as': multiplier * (-1.0/pow(dy, 2)),
                'b': 0.0
            })
    
    def _compute_interior(self, block_id, solver, field):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
            solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])
            solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
            solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])
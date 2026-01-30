from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType

class SecondOrderBackwards(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, runtime, var_name, multiplier):
        dt = runtime.dt
        if runtime.current_timestep > 1:
            for block_id in range(0, self.mesh.num_blocks):
                self.coefficients.append({
                    'ap': 3.0 / (2.0 * dt) * multiplier,
                    'b':  [2.0 / dt * multiplier, -1.0 / (2.0 * dt) * multiplier],
                    'is_second_order': True
                })
        else:
            for block_id in range(0, self.mesh.num_blocks):
                self.coefficients.append({
                    'ap': 1.0/dt * multiplier,
                    'b':  [1.0/dt * multiplier, 0.0],
                    'is_second_order': False
                })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ap_value = self.coefficients[block_id]['ap']
            solver.add_to_A(ap_index, ap_index, ap_value)

            b_index = self.mesh.map3Dto1D(block_id, i, j)
            b_value = self.coefficients[block_id]['b'][0] * self.field_manager.fields[var_name].old[block_id, i, j] + \
                self.coefficients[block_id]['b'][1] * self.field_manager.fields[var_name].oldold[block_id, i, j]
            solver.add_to_b(b_index, b_value)

    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        block1, i1, j1 = ij
        return self.coefficients[block1]['b'][0] * self.field_manager.fields[var_name].old[block1, i1, j1] + \
                self.coefficients[block1]['b'][1] * self.field_manager.fields[var_name].oldold[block1, i1, j1]
    
    def _east_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                bc_value = self.coefficients[block_id]['b'][0] * self.field_manager.fields[var_name].old[block_id, i, j] + \
                    self.coefficients[block_id]['b'][1] * self.field_manager.fields[var_name].oldold[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _west_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                bc_value = self.coefficients[block_id]['b'][0] * self.field_manager.fields[var_name].old[block_id, i, j] + \
                    self.coefficients[block_id]['b'][1] * self.field_manager.fields[var_name].oldold[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _north_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                bc_value = self.coefficients[block_id]['b'][0] * self.field_manager.fields[var_name].old[block_id, i, j] + \
                    self.coefficients[block_id]['b'][1] * self.field_manager.fields[var_name].oldold[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _south_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                bc_value = self.coefficients[block_id]['b'][0] * self.field_manager.fields[var_name].old[block_id, i, j] + \
                    self.coefficients[block_id]['b'][1] * self.field_manager.fields[var_name].oldold[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)
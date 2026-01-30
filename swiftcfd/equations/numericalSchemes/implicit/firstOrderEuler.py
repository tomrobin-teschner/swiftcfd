from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType

class FirstOrderEuler(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, runtime, var_name, multiplier):
        dt = runtime.dt
        for block_id in range(0, self.mesh.num_blocks):
            self.coefficients.append({
                'ap': 1.0/dt * multiplier,
                'b':  1.0/dt * multiplier
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
        return multiplier * self.field_manager.fields[var_name].old[block1, i1, j1]

    def _east_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
        #     bc_value = self.bc.get_bc_value(block_id, "east")
        #     bc_coefficient = 1.0

        #     for (i, j) in self.mesh.loop_east(block_id, 1):
        #         ap_index = self.mesh.map3Dto1D(block_id, i, j)
        #         solver.add_to_A(ap_index, ap_index, bc_coefficient)
        #         solver.add_to_b(ap_index, bc_value)
        # else:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _west_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
        #     bc_value = self.bc.get_bc_value(block_id, "west")
        #     bc_coefficient = 1.0

        #     for (i, j) in self.mesh.loop_west(block_id, 1):
        #         ap_index = self.mesh.map3Dto1D(block_id, i, j)
        #         solver.add_to_A(ap_index, ap_index, bc_coefficient)
        #         solver.add_to_b(ap_index, bc_value)
        # else:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _north_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
        #     bc_value = self.bc.get_bc_value(block_id, "north")
        #     bc_coefficient = 1.0

        #     for (i, j) in self.mesh.loop_north(block_id, 1):
        #         ap_index = self.mesh.map3Dto1D(block_id, i, j)
        #         solver.add_to_A(ap_index, ap_index, bc_coefficient)
        #         solver.add_to_b(ap_index, bc_value)
        # else:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

    def _south_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
        #     bc_value = self.bc.get_bc_value(block_id, "south")
        #     bc_coefficient = 1.0

        #     for (i, j) in self.mesh.loop_south(block_id, 1):
        #         ap_index = self.mesh.map3Dto1D(block_id, i, j)
        #         solver.add_to_A(ap_index, ap_index, bc_coefficient)
        #         solver.add_to_b(ap_index, bc_value)
        # else:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)

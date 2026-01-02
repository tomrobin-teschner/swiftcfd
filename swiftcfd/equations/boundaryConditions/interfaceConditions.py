from swiftcfd.equations.boundaryConditions.boundaryConditions import BCType
from swiftcfd.equations.boundaryConditions.cornerPoint import CornerPoint

class InterfaceConditions():
    def __init__(self, mesh, bc):
        self.mesh = mesh
        self.bc = bc
        self.corner_points = CornerPoint(mesh, bc)

    def apply_interface_conditions(self, block_id, solver, field, scheme):      
        # check for east interface            
        if self.bc.bc_type[block_id]["east"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["east"])
            east_index = 1

            for (i, j) in self.mesh.loop_east(block_id, 1):
                ij = (block_id, i, j)
                ip1j = (neighbour_id, east_index, j)
                im1j = (block_id, i - 1, j)
                ijp1 = (block_id, i, j + 1)
                ijm1 = (block_id, i, j - 1)

                ap_index = self.mesh.map3Dto1D(*ij)
                ae_index = self.mesh.map3Dto1D(*ip1j)
                aw_index = self.mesh.map3Dto1D(*im1j)
                an_index = self.mesh.map3Dto1D(*ijp1)
                as_index = self.mesh.map3Dto1D(*ijm1)
                b_index  = self.mesh.map3Dto1D(*ij)

                if 'ap' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ap_index, scheme.coefficients[block_id]['ap'])
                if 'ae' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ae_index, scheme.coefficients[neighbour_id]['ae'])
                if 'aw' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, aw_index, scheme.coefficients[block_id]['aw'])
                if 'an' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, an_index, scheme.coefficients[block_id]['an'])
                if 'as' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, as_index, scheme.coefficients[block_id]['as'])
                if 'b' in scheme.coefficients[block_id]:
                    solver.add_to_b(b_index, scheme.get_right_hand_side_contribution(ij, ip1j, im1j, ijp1, ijm1, field))

        # check for west interface
        if self.bc.bc_type[block_id]["west"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["west"])
            west_index = int(self.mesh.num_x[neighbour_id] - 2)

            for (i, j) in self.mesh.loop_west(block_id, 1):
                ij = (block_id, i, j)
                ip1j = (block_id, i + 1, j)
                im1j = (neighbour_id, west_index, j)
                ijp1 = (block_id, i, j + 1)
                ijm1 = (block_id, i, j - 1)

                ap_index = self.mesh.map3Dto1D(*ij)
                ae_index = self.mesh.map3Dto1D(*ip1j)
                aw_index = self.mesh.map3Dto1D(*im1j)
                an_index = self.mesh.map3Dto1D(*ijp1)
                as_index = self.mesh.map3Dto1D(*ijm1)
                b_index  = self.mesh.map3Dto1D(*ij)

                if 'ap' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ap_index, scheme.coefficients[block_id]['ap'])
                if 'ae' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ae_index, scheme.coefficients[block_id]['ae'])
                if 'aw' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, aw_index, scheme.coefficients[neighbour_id]['aw'])
                if 'an' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, an_index, scheme.coefficients[block_id]['an'])
                if 'as' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, as_index, scheme.coefficients[block_id]['as'])
                if 'b' in scheme.coefficients[block_id]:
                    solver.add_to_b(b_index, scheme.get_right_hand_side_contribution(ij, ip1j, im1j, ijp1, ijm1, field))
                            
        # check for north interface
        if self.bc.bc_type[block_id]["north"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["north"])
            north_index = 1
                
            for (i, j) in self.mesh.loop_north(block_id, 1):
                ij = (block_id, i, j)
                ip1j = (block_id, i + 1, j)
                im1j = (block_id, i - 1, j)
                ijp1 = (neighbour_id, i, north_index)
                ijm1 = (block_id, i, j - 1)

                ap_index = self.mesh.map3Dto1D(*ij)
                ae_index = self.mesh.map3Dto1D(*ip1j)
                aw_index = self.mesh.map3Dto1D(*im1j)
                an_index = self.mesh.map3Dto1D(*ijp1)
                as_index = self.mesh.map3Dto1D(*ijm1)
                b_index  = self.mesh.map3Dto1D(*ij)

                if 'ap' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ap_index, scheme.coefficients[block_id]['ap'])
                if 'ae' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ae_index, scheme.coefficients[block_id]['ae'])
                if 'aw' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, aw_index, scheme.coefficients[block_id]['aw'])
                if 'an' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, an_index, scheme.coefficients[block_id]['an'])
                if 'as' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, as_index, scheme.coefficients[block_id]['as'])
                if 'b' in scheme.coefficients[block_id]:
                    solver.add_to_b(b_index, scheme.get_right_hand_side_contribution(ij, ip1j, im1j, ijp1, ijm1, field))

        # check for south interface
        if self.bc.bc_type[block_id]["south"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["south"])
            south_index = int(self.mesh.num_y[neighbour_id] - 2)

            for (i, j) in self.mesh.loop_south(block_id, 1):
                ij = (block_id, i, j)
                ip1j = (block_id, i + 1, j)
                im1j = (block_id, i - 1, j)
                ijp1 = (block_id, i, j + 1)
                ijm1 = (neighbour_id, i, south_index)

                ap_index = self.mesh.map3Dto1D(*ij)
                ae_index = self.mesh.map3Dto1D(*ip1j)
                aw_index = self.mesh.map3Dto1D(*im1j)
                an_index = self.mesh.map3Dto1D(*ijp1)
                as_index = self.mesh.map3Dto1D(*ijm1)
                b_index  = self.mesh.map3Dto1D(*ij)

                if 'ap' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ap_index, scheme.coefficients[block_id]['ap'])
                if 'ae' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, ae_index, scheme.coefficients[block_id]['ae'])
                if 'aw' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, aw_index, scheme.coefficients[block_id]['aw'])
                if 'an' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, an_index, scheme.coefficients[block_id]['an'])
                if 'as' in scheme.coefficients[block_id]:
                    solver.add_to_A(ap_index, as_index, scheme.coefficients[block_id]['as'])
                if 'b' in scheme.coefficients[block_id]:
                    solver.add_to_b(b_index, scheme.get_right_hand_side_contribution(ij, ip1j, im1j, ijp1, ijm1, field))
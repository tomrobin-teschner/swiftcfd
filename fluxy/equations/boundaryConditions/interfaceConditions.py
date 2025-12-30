from fluxy.equations.boundaryConditions.boundaryConditions import BCType
from fluxy.equations.boundaryConditions.cornerPoint import CornerPoint

class InterfaceConditions():
    def __init__(self, params, mesh, bc):
        self.params = params
        self.mesh = mesh
        self.bc = bc
        self.corner_points = CornerPoint(mesh, bc)

    def apply_interface_conditions(self, block_id, solver, field, coefficients):      
        # check for east interface            
        if self.bc.bc_type[block_id]["east"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["east"])
            east_index = 1

            for (i, j) in self.mesh.loop_east(block_id, 1):
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                ae_index = self.mesh.map3Dto1D(neighbour_id, east_index, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
                b_index  = self.mesh.map3Dto1D(block_id, i, j)

                solver.add_to_A(ap_index, ap_index, coefficients[block_id]['ap'])
                solver.add_to_A(ap_index, ae_index, coefficients[neighbour_id]['ae'])
                solver.add_to_A(ap_index, aw_index, coefficients[block_id]['aw'])
                solver.add_to_A(ap_index, an_index, coefficients[block_id]['an'])
                solver.add_to_A(ap_index, as_index, coefficients[block_id]['as'])
                solver.add_to_b(b_index, coefficients[block_id]['b'] * field.old[block_id, i, j])

        # check for west interface
        if self.bc.bc_type[block_id]["west"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["west"])
            west_index = int(self.mesh.num_x[neighbour_id] - 2)

            for (i, j) in self.mesh.loop_west(block_id, 1):
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(neighbour_id, west_index, j)
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
                b_index  = self.mesh.map3Dto1D(block_id, i, j)

                solver.add_to_A(ap_index, ap_index, coefficients[block_id]['ap'])
                solver.add_to_A(ap_index, ae_index, coefficients[block_id]['ae'])
                solver.add_to_A(ap_index, aw_index, coefficients[neighbour_id]['aw'])
                solver.add_to_A(ap_index, an_index, coefficients[block_id]['an'])
                solver.add_to_A(ap_index, as_index, coefficients[block_id]['as'])
                solver.add_to_b(b_index, coefficients[block_id]['b'] * field.old[block_id, i, j])
                            
        # check for north interface
        if self.bc.bc_type[block_id]["north"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["north"])
            north_index = 1
                
            for (i, j) in self.mesh.loop_north(block_id, 1):
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
                an_index = self.mesh.map3Dto1D(neighbour_id, i, north_index)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
                b_index  = self.mesh.map3Dto1D(block_id, i, j)

                solver.add_to_A(ap_index, ap_index, coefficients[block_id]['ap'])
                solver.add_to_A(ap_index, ae_index, coefficients[block_id]['ae'])
                solver.add_to_A(ap_index, aw_index, coefficients[block_id]['aw'])
                solver.add_to_A(ap_index, an_index, coefficients[block_id]['an'])
                solver.add_to_A(ap_index, as_index, coefficients[block_id]['as'])
                solver.add_to_b(b_index, coefficients[block_id]['b'] * field.old[block_id, i, j])

        # check for south interface
        if self.bc.bc_type[block_id]["south"] == BCType.interface:
            neighbour_id = int(self.bc.bc_value[block_id]["south"])
            south_index = int(self.mesh.num_y[neighbour_id] - 2)

            for (i, j) in self.mesh.loop_south(block_id, 1):
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(neighbour_id, i, south_index)
                b_index  = self.mesh.map3Dto1D(block_id, i, j)

                solver.add_to_A(ap_index, ap_index, coefficients[block_id]['ap'])
                solver.add_to_A(ap_index, ae_index, coefficients[block_id]['ae'])
                solver.add_to_A(ap_index, aw_index, coefficients[block_id]['aw'])
                solver.add_to_A(ap_index, an_index, coefficients[block_id]['an'])
                solver.add_to_A(ap_index, as_index, coefficients[block_id]['as'])
                solver.add_to_b(b_index, coefficients[block_id]['b'] * field.old[block_id, i, j])
        
        # finally, check if corner points are present, and if so, add them to the system
        for corner in self.corner_points.internal_corner_points:
            solver.add_to_A(corner['index']['ap'], corner['index']['ap'], coefficients[corner['block']['ap']]['ap'])
            solver.add_to_A(corner['index']['ap'], corner['index']['ae'], coefficients[corner['block']['ae']]['ae'])
            solver.add_to_A(corner['index']['ap'], corner['index']['aw'], coefficients[corner['block']['aw']]['aw'])
            solver.add_to_A(corner['index']['ap'], corner['index']['an'], coefficients[corner['block']['an']]['an'])
            solver.add_to_A(corner['index']['ap'], corner['index']['as'], coefficients[corner['block']['as']]['as'])
            solver.add_to_b(corner['index']['b'], coefficients[corner['block']['b']]['b'] * field.old[corner['block']['ap'], i, j])

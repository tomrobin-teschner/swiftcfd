from enum import Enum, auto
from fluxy.equations.boundaryConditions.boundaryConditions import BCType

class CornerType(Enum):
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    TOP_LEFT = auto()
    TOP_RIGHT = auto()

class CornerPoint:
    def __init__(self, mesh, bc):
        self.mesh = mesh
        self.bc = bc
        self.internal_corner_points = []

        self._find_internal_corner_points()

    def _find_internal_corner_points(self):
        for block in range(0, self.mesh.num_blocks):
            east_bc  = self.bc.bc_type[block]["east"]
            west_bc  = self.bc.bc_type[block]["west"]
            north_bc = self.bc.bc_type[block]["north"]
            south_bc = self.bc.bc_type[block]["south"] 

            east_block  = self.bc.bc_value[block]["east"]
            west_block  = self.bc.bc_value[block]["west"]
            north_block = self.bc.bc_value[block]["north"]
            south_block = self.bc.bc_value[block]["south"]

            # check bottom left corner
            if west_bc == BCType.interface and south_bc == BCType.interface:
                south_block_bc = self.bc.bc_type[west_block]["south"]
                west_block_bc  = self.bc.bc_type[south_block]["west"]

                # evaluates to true if internal corner poitn has been found
                if south_block_bc == BCType.interface and west_block_bc == BCType.interface:
                    south_west_block = self.bc.bc_value[west_block]["south"]

                    ap_block = block
                    ae_block = block
                    aw_block = south_west_block
                    an_block = block
                    as_block = south_west_block
                    b_block  = block

                    ap_index = self.mesh.map3Dto1D(ap_block, 0, 0)
                    ae_index = self.mesh.map3Dto1D(ae_block, 1, 0)
                    aw_index = self.mesh.map3Dto1D(aw_block, self.mesh.num_x[aw_block] - 2, self.mesh.num_y[aw_block] - 1)
                    an_index = self.mesh.map3Dto1D(an_block, 0, 1)
                    as_index = self.mesh.map3Dto1D(as_block, self.mesh.num_x[as_block] - 1, self.mesh.num_y[as_block] - 2)
                    b_index  = self.mesh.map3Dto1D(b_block, 0, 0)

                    i, j = 0, 0

                    corner_point = {
                        "index": {
                            "ap": ap_index, "ae": ae_index, "aw": aw_index, "an": an_index, "as": as_index, "b":  b_index, "i": i, "j": j
                        },
                        "block": {
                            "ap": ap_block, "ae": ae_block, "aw": aw_block, "an": an_block, "as": as_block, "b":  b_block
                        }

                    }

                    self.internal_corner_points.append(corner_point)
            
            # check bottom right corner
            if east_bc == BCType.interface and south_bc == BCType.interface:
                south_block_bc = self.bc.bc_type[east_block]["south"]
                east_block_bc  = self.bc.bc_type[south_block]["east"]

                # evaluates to true if internal corner poitn has been found
                if south_block_bc == BCType.interface and east_block_bc == BCType.interface:
                    south_east_block = self.bc.bc_value[east_block]["south"]

                    ap_block = block
                    ae_block = south_east_block
                    aw_block = block
                    an_block = block
                    as_block = south_east_block
                    b_block  = block

                    ap_index = self.mesh.map3Dto1D(ap_block, self.mesh.num_x[ap_block] - 1, 0)
                    ae_index = self.mesh.map3Dto1D(ae_block, 1, self.mesh.num_y[ae_block] - 1)
                    aw_index = self.mesh.map3Dto1D(aw_block, self.mesh.num_x[aw_block] - 2, 0)
                    an_index = self.mesh.map3Dto1D(an_block, self.mesh.num_x[an_block] - 1, 1)
                    as_index = self.mesh.map3Dto1D(as_block, 0, self.mesh.num_y[as_block] - 2)
                    b_index  = self.mesh.map3Dto1D(b_block, self.mesh.num_x[b_block] - 1, 0)

                    i, j = self.mesh.num_x[ap_block] - 1, 0

                    corner_point = {
                        "index": {
                            "ap": ap_index, "ae": ae_index, "aw": aw_index, "an": an_index, "as": as_index, "b":  b_index, "i": i, "j": j
                        },
                        "block": {
                            "ap": ap_block, "ae": ae_block, "aw": aw_block, "an": an_block, "as": as_block, "b":  b_block
                        }
                    }

                    self.internal_corner_points.append(corner_point)
            
            # check top left corner
            if west_bc == BCType.interface and north_bc == BCType.interface:
                north_block_bc = self.bc.bc_type[west_block]["north"]
                west_block_bc  = self.bc.bc_type[north_block]["west"]

                # evaluates to true if internal corner poitn has been found
                if north_block_bc == BCType.interface and west_block_bc == BCType.interface:
                    north_west_block = self.bc.bc_value[west_block]["north"]

                    ap_block = block
                    ae_block = block
                    aw_block = north_west_block
                    an_block = north_west_block
                    as_block = block
                    b_block  = block

                    ap_index = self.mesh.map3Dto1D(ap_block, 0, self.mesh.num_x[ap_block] - 1)
                    ae_index = self.mesh.map3Dto1D(ae_block, 1, self.mesh.num_x[ae_block] - 1)
                    aw_index = self.mesh.map3Dto1D(aw_block, self.mesh.num_x[aw_block] - 2, 0)
                    an_index = self.mesh.map3Dto1D(an_block, self.mesh.num_x[an_block] - 1, 1)
                    as_index = self.mesh.map3Dto1D(as_block, 0, self.mesh.num_x[as_block] - 2)
                    b_index  = self.mesh.map3Dto1D(b_block, 0, self.mesh.num_x[b_block] - 1)

                    i, j = 0, self.mesh.num_x[ap_block] - 1

                    corner_point = {
                        "index": {
                            "ap": ap_index, "ae": ae_index, "aw": aw_index, "an": an_index, "as": as_index, "b":  b_index, "i": i, "j": j
                        },
                        "block": {
                            "ap": ap_block, "ae": ae_block, "aw": aw_block, "an": an_block, "as": as_block, "b":  b_block
                        }
                    }

                    self.internal_corner_points.append(corner_point)

            # check top right corner
            if east_bc == BCType.interface and north_bc == BCType.interface:
                north_block_bc = self.bc.bc_type[east_block]["north"]
                east_block_bc  = self.bc.bc_type[north_block]["east"]

                # evaluates to true if internal corner poitn has been found
                if north_block_bc == BCType.interface and east_block_bc == BCType.interface:
                    north_east_block = self.bc.bc_value[east_block]["north"]

                    ap_block = block
                    ae_block = north_east_block
                    aw_block = block
                    an_block = north_east_block
                    as_block = block
                    b_block  = block

                    ap_index = self.mesh.map3Dto1D(ap_block, self.mesh.num_x[ap_block] - 1, self.mesh.num_y[ap_block] - 1)
                    ae_index = self.mesh.map3Dto1D(ae_block, 1, 0)
                    aw_index = self.mesh.map3Dto1D(aw_block, self.mesh.num_x[aw_block] - 2, self.mesh.num_y[aw_block] - 1)
                    an_index = self.mesh.map3Dto1D(an_block, 0, 1)
                    as_index = self.mesh.map3Dto1D(as_block, self.mesh.num_x[as_block] - 1, self.mesh.num_y[as_block] - 2)
                    b_index  = self.mesh.map3Dto1D(b_block, self.mesh.num_x[ap_block] - 1, self.mesh.num_y[ap_block] - 1)

                    i, j = self.mesh.num_x[ap_block] - 1, self.mesh.num_y[ap_block] - 1

                    corner_point = {
                        "index": {
                            "ap": ap_index, "ae": ae_index, "aw": aw_index, "an": an_index, "as": as_index, "b":  b_index, "i": i, "j": j
                        },
                        "block": {
                            "ap": ap_block, "ae": ae_block, "aw": aw_block, "an": an_block, "as": as_block, "b":  b_block
                        }
                    }

                    self.internal_corner_points.append(corner_point)

    def average_field_at_corner_point(self, field):
        if len(self.internal_corner_points) > 0:
            ap_index = [d["index"]["ap"] for d in self.internal_corner_points]
            i_index = [d["index"]["i"] for d in self.internal_corner_points]
            j_index = [d["index"]["j"] for d in self.internal_corner_points]
            block = [d["block"]["ap"] for d in self.internal_corner_points]

            assert(len(ap_index) % 4 == 0)

            ap_sorted, i_sorted, j_sorted, block_sorted = zip(*sorted(zip(ap_index, i_index, j_index, block)))
            ap_sorted = list(ap_sorted)
            i_sorted = list(i_sorted)
            j_sorted = list(j_sorted)
            block_sorted = list(block_sorted)

            for index in range(0, len(ap_sorted), 4):
                b1, i1, j1 = block_sorted[index + 0], i_sorted[index + 0], j_sorted[index + 0]
                b2, i2, j2 = block_sorted[index + 1], i_sorted[index + 1], j_sorted[index + 1]
                b3, i3, j3 = block_sorted[index + 2], i_sorted[index + 2], j_sorted[index + 2]
                b4, i4, j4 = block_sorted[index + 3], i_sorted[index + 3], j_sorted[index + 3]

                phi1 = field[b1, i1, j1]
                phi2 = field[b2, i2, j2]
                phi3 = field[b3, i3, j3]
                phi4 = field[b4, i4, j4]

                phi_avg = (phi1 + phi2 + phi3 + phi4) / 4

                field[b1, i1, j1] = phi_avg
                field[b2, i2, j2] = phi_avg
                field[b3, i3, j3] = phi_avg
                field[b4, i4, j4] = phi_avg
from swiftcfd.field.field import Field
class FirstOrderGradient:
    def __init__(self, mesh, field_manager, var_name):
        self.mesh = mesh
        self.field_manager = field_manager
        self.var_name = var_name
        self.x = Field(self.mesh, self.var_name)
        self.y = Field(self.mesh, self.var_name)

    def compute(self):
        # compute gradients for internal mesh points only
        for block in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block)
            for (i, j) in self.mesh.internal_loop_single_block(block):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

                self.x[block, i, j] = (phi_ip1 - phi_im1) / (2.0 * dx)
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / (2.0 * dy)
        
        # compute gradient on boundaries / interfaces using one-sided finite differences
        for block in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block)
            for (i, j) in self.mesh.loop_east(block, 1):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

                self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / (2.0 * dy)

            for (i, j) in self.mesh.loop_west(block, 1):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i, j]

                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

                self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / (2.0 * dy)

            for (i, j) in self.mesh.loop_north(block, 1):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

                self.x[block, i, j] = (phi_ip1 - phi_im1) / (2.0 * dx)
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

            for (i, j) in self.mesh.loop_south(block, 1):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j]

                self.x[block, i, j] = (phi_ip1 - phi_im1) / (2.0 * dx)
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

        # compute gradients at corner points
        for block in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block)

            # bottom left
            i = 0
            j = 0

            phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
            phi_im1 = self.field_manager.fields[self.var_name][block, i, j]

            phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
            phi_jm1 = self.field_manager.fields[self.var_name][block, i, j]

            self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
            self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

            # bottom right
            i = self.mesh.num_x[block] - 1
            j = 0

            phi_ip1 = self.field_manager.fields[self.var_name][block, i, j]
            phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

            phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
            phi_jm1 = self.field_manager.fields[self.var_name][block, i, j]

            self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
            self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

            # top left
            i = 0
            j = self.mesh.num_y[block] - 1

            phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
            phi_im1 = self.field_manager.fields[self.var_name][block, i, j]

            phi_jp1 = self.field_manager.fields[self.var_name][block, i, j]
            phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

            self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
            self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

            # top right
            i = self.mesh.num_x[block] - 1
            j = self.mesh.num_y[block] - 1

            phi_ip1 = self.field_manager.fields[self.var_name][block, i, j]
            phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]

            phi_jp1 = self.field_manager.fields[self.var_name][block, i, j]
            phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]

            self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
            self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

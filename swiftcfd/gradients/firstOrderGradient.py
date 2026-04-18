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

            # central difference in x on internal cells
            for (i, j) in self.mesh.loop_cells_with_offset(block, 1, 0):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]
                self.x[block, i, j] = (phi_ip1 - phi_im1) / (2.0 * dx)

            # one-sided gradient at west boundary boundaries
            for (i, j) in self.mesh.loop_west_bc(block):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i + 1, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i, j]
                self.x[block, i, j] = (phi_ip1 - phi_im1) / dx

            # one-sided gradient at east boundary boundaries
            for (i, j) in self.mesh.loop_east_bc(block):
                phi_ip1 = self.field_manager.fields[self.var_name][block, i, j]
                phi_im1 = self.field_manager.fields[self.var_name][block, i - 1, j]
                self.x[block, i, j] = (phi_ip1 - phi_im1) / dx
            
            # sweep over all cells in y, ignore cells at east/west boundary
            for (i, j) in self.mesh.loop_cells_with_offset(block, 0, 1):
                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / (2.0 * dy)
            
            # one-sided gradient at south boundary boundaries
            for (i, j) in self.mesh.loop_south_bc(block):
                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j + 1]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j]
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

            # one-sided gradient at north boundary boundaries
            for (i, j) in self.mesh.loop_north_bc(block):
                phi_jp1 = self.field_manager.fields[self.var_name][block, i, j]
                phi_jm1 = self.field_manager.fields[self.var_name][block, i, j - 1]
                self.y[block, i, j] = (phi_jp1 - phi_jm1) / dy

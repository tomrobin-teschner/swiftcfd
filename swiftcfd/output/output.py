from os.path import join, exists
from os import listdir, remove
from os import mkdir

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

class Output():
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager
        self.case = self.params('solver', 'output', 'filename')
        self.writing_frequency = self.params('solver', 'output', 'writingFrequency')
        self.out_folder = join('output', self.case)

        # create output folder if not exists and delete old files
        if not exists(self.out_folder):
            mkdir(self.out_folder)

    def write_tecplot_file(self, iteration = -1):
        filename = self.case

        if iteration == -1:
            filename += '.dat' 
            self._write_tecplot(self.case, filename)
        else:
            if iteration % self.writing_frequency == 0:
                filename += f'_{iteration:010d}.dat' 
                self._write_tecplot(self.case, filename)

    def _write_tecplot(self, case, filename):
        with open(join(self.out_folder, filename), 'w') as f:
            f.write(f'TITLE = "{case}"\n')
            f.write('VARIABLES = "x", "y"')
            for key, _ in self.field_manager.fields.items():
                f.write(f', "{key}"')
            f.write('\n')

            for block in range(0, self.mesh.num_blocks):
                f.write(f'\nZONE T="Block{block + 1}", I={self.mesh.num_x[block]}, J={self.mesh.num_y[block]}, F=POINT\n')
                for j in range(0, self.mesh.num_y[block]):
                    for i in range(0, self.mesh.num_x[block]):
                        f.write(f'{self.mesh.x[block][i][j]} {self.mesh.y[block][i][j]}')
                        for _, value in self.field_manager.fields.items():
                            f.write(f' {value[block, i, j]}')
                        f.write('\n')
    
    def plot_contours(self):
        n_fields = len(self.field_manager.fields)

        fig, axes = plt.subplots(
            n_fields, 1,
            figsize=(8, 2.25 * n_fields),
            sharex=True
        )

        # In case there is only one field
        if n_fields == 1:
            axes = [axes]

        for ax, (field_name, field_data) in zip(axes, self.field_manager.fields.items()):
            vmin = np.inf
            vmax = -np.inf

            # ---- first pass: get min/max for this subplot ----
            for block in range(self.mesh.num_blocks):
                nx = self.mesh.num_x[block]
                ny = self.mesh.num_y[block]
                offset = self.mesh.points_offset[block]

                field = field_data._data[offset : offset + nx * ny].reshape(ny, nx)
                vmin = min(vmin, field.min())
                vmax = max(vmax, field.max())

            # ---- second pass: plotting all blocks on the same subplot ----
            cs = None
            for block in range(self.mesh.num_blocks):
                nx = self.mesh.num_x[block]
                ny = self.mesh.num_y[block]
                offset = self.mesh.points_offset[block]

                field = field_data._data[offset : offset + nx * ny].reshape(ny, nx)
                x = self.mesh.x[block].T
                y = self.mesh.y[block].T

                cs = ax.contourf(
                    x, y, field,
                    levels=20,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="jet"
                )

            ax.set_aspect("equal")
            ax.set_ylabel(field_name)

            # ---- colorbar per subplot ----
            fig.colorbar(
                cs,          # link to the last QuadContourSet of this subplot
                ax=ax,
                orientation="horizontal",
                location="top",
                shrink=0.8,
                pad=0.05
            )

        axes[-1].set_xlabel("x")

        fig.tight_layout()
        fig.savefig(
            join(self.out_folder, "contours.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    def plot_residuals(self):
        df = pd.read_csv(join(self.out_folder, 'residuals.csv'))

        plt.style.use('bmh')
        fig, ax = plt.subplots()

        for var_name in df.columns:
            ax.plot(df.index, df[var_name], label=var_name)

        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            join(self.out_folder, "residuals.png"),
            dpi=300,
            bbox_inches="tight"
        )  

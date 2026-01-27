from os.path import join, exists
from os import listdir, remove
from os import mkdir

from swiftcfd.enums import PrimitiveVariables as pv

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

    def write(self, iteration = -1):
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

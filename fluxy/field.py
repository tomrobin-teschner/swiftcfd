import numpy as np

class Field():
    def __init__(self, params, mesh, name):
        self.params = params
        self.mesh = mesh
        self.name = name

        self.data = []

        for block in range(0, self.mesh.num_blocks):
            num_x = self.mesh.num_x[block]
            num_y = self.mesh.num_y[block]
            self.data.append(np.zeros((num_x, num_y)))
    
    def __getitem__(self, block):
        return self.data[block]
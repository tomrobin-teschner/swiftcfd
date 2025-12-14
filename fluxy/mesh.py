import numpy as np

class Mesh():
    def __init__(self, params):
        self.params = params
        self.num_blocks = len(self.params.mesh())

        self.x_start = np.zeros(self.num_blocks)
        self.x_end = np.zeros(self.num_blocks)
        self.y_start = np.zeros(self.num_blocks)
        self.y_end = np.zeros(self.num_blocks)
        self.num_x = np.zeros(self.num_blocks, dtype = int)
        self.num_y = np.zeros(self.num_blocks, dtype = int)
        self.points_offset = np.zeros(self.num_blocks, dtype = int)

        self.total_points = 0
        self.points_per_block = np.zeros(self.num_blocks, dtype = int)

        for i in range(0, self.num_blocks):
            block = f'block{i + 1}'
            self.x_start[i] = self.params.mesh(block, 'x', 'start')
            self.x_end[i] = self.params.mesh(block, 'x', 'end')
            self.y_start[i] = self.params.mesh(block, 'y', 'start')
            self.y_end[i] = self.params.mesh(block, 'y', 'end')
            self.num_x[i] = int(self.params.mesh(block, 'x', 'num'))
            self.num_y[i] = int(self.params.mesh(block, 'y', 'num'))

            self.points_per_block[i] = self.num_x[i] * self.num_y[i]
            self.total_points += self.points_per_block[i]

            if i == 0:
                self.points_offset[i] = 0
            else:
                self.points_offset[i] = self.points_offset[i - 1] + int(self.num_x[i - 1] * self.num_y[i - 1])
            
        self.x = []
        self.y = []

    def map3Dto1D(self, block_id, i, j):
        offset = self.points_offset[block_id]
        stride = int(self.num_x[block_id]) * j
        return offset + stride + i        

    def create(self):
        for block in range(0, self.num_blocks):
            x_start = self.x_start[block]
            x_end = self.x_end[block]
            
            y_start = self.y_start[block]
            y_end = self.y_end[block]
            
            num_x = self.num_x[block]
            num_y = self.num_y[block]

            x = np.zeros((num_x, num_y))
            y = np.zeros((num_x, num_y))

            for i in range(0, num_x):
                for j in range(0, num_y):
                    x[i][j] = x_start + i * (x_end - x_start) / (num_x - 1)
                    y[i][j] = y_start + j * (y_end - y_start) / (num_y - 1)

            self.x.append(x)
            self.y.append(y)
    
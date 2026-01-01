from petsc4py import PETSc
from petsc4py import init as petsc_init

import numpy as np
import matplotlib.pyplot as plt

from swiftcfd.equations.linearAlgebraSolver.solverFactory import SolverFactory

class LinearAlgebraSolver():
    def __init__(self, params, mesh, var_name):
        # total points in mesh
        self.total_points = mesh.total_points
        self.mesh = mesh
        self.var_name = var_name

        # initialise petsc for usage
        petsc_init()

        # create coefficient matrix A
        self.A = PETSc.Mat().create()
        self.A.setSizes([self.total_points, self.total_points])
        self.A.setType(PETSc.Mat.Type.SEQAIJ)
        self.A.setPreallocationNNZ(5)
        self.A.setUp()

        # create right-hand side vector
        self.b = PETSc.Vec().createSeq(self.total_points)

        # create linear solver
        self.ksp = SolverFactory().create(params, self.var_name)
        self.ksp.setOperators(self.A)        
    
    def reset_A(self):
        self.A.zeroEntries()
    
    def reset_b(self):
        self.b.zeroEntries()

    def add_to_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.INSERT_VALUES)
    
    def add_to_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.INSERT_VALUES)

    def assemble(self):
        self.A.assemblyBegin()
        self.A.assemblyEnd()

        self.b.assemblyBegin()
        self.b.assemblyEnd()

    def field_to_petsc_vec(self, field):
        vec = PETSc.Vec().createWithArray(field._data)
        return vec

    def solve(self, field):
        field_petsc = self.field_to_petsc_vec(field)
        self.ksp.solve(self.b, field_petsc)

    def get_solver_statistics(self):
        num_iterations = self.ksp.getIterationNumber()
        res_norm = self.ksp.getResidualNorm()
        has_converged = self.ksp.getConvergedReason() >= 0
        return num_iterations, res_norm, has_converged
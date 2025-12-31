import unittest
from petsc4py import PETSc
import numpy as np

from swiftcfd.equations.linearAlgebraSolver.linearAlgebraSolver import LinearAlgebraSolver
from swiftcfd.field import Field

class TestLinearAlgebraSolver(unittest.TestCase):
    def setUp(self):
        class FakeMesh():
            def __init__(self, total_points):
                self.total_points = total_points
        
        mesh = FakeMesh(2)
        self.solver = LinearAlgebraSolver(None, mesh)

    def test_inserting_into_A(self):
        # arrange
        self.solver.insert_into_A(0, 0, 1)
        self.solver.insert_into_A(0, 1, 2)
        self.solver.insert_into_A(1, 0, 3)
        self.solver.insert_into_A(1, 1, 4)

        # act
        self.solver.assemble()

        # assert
        self.assertEqual(self.solver.A.getValues(0, 0), 1)
        self.assertEqual(self.solver.A.getValues(0, 1), 2)
        self.assertEqual(self.solver.A.getValues(1, 0), 3)
        self.assertEqual(self.solver.A.getValues(1, 1), 4)

    def test_adding_to_A(self):
        # arrange
        self.solver.add_to_A(0, 0, 1)
        self.solver.add_to_A(0, 0, 2)
        self.solver.add_to_A(0, 0, 3.5)
        
        # act
        self.solver.assemble()

        # assert
        self.assertEqual(self.solver.A.getValues(0, 0), 6.5)
    
    def test_insert_into_b(self):
        # arrange
        self.solver.insert_into_b(0, 1)
        self.solver.insert_into_b(1, 2.7)

        # act
        self.solver.assemble()

        # assert
        self.assertEqual(self.solver.b.getValues(0), 1)
        self.assertEqual(self.solver.b.getValues(1), 2.7)
    
    def test_add_to_b(self):
        # arrange
        self.solver.add_to_b(0, 1)
        self.solver.add_to_b(0, 2.7)

        # act
        self.solver.assemble()

        # assert
        self.assertEqual(self.solver.b.getValues(0), 3.7)
    
    def test_reset_A(self):
        # arrange
        self.solver.insert_into_A(0, 0, 1)
        self.solver.insert_into_A(0, 1, 2)
        self.solver.insert_into_A(1, 0, 3)
        self.solver.insert_into_A(1, 1, 4)
        self.solver.assemble()

        # act
        self.solver.reset_A()

        # assert
        self.assertEqual(self.solver.A.getValues(0, 0), 0)
        self.assertEqual(self.solver.A.getValues(0, 1), 0)
        self.assertEqual(self.solver.A.getValues(1, 0), 0)
        self.assertEqual(self.solver.A.getValues(1, 1), 0)

    def test_reset_b(self):
        # arrange
        self.solver.insert_into_b(0, 1)
        self.solver.insert_into_b(1, 2.7)
        self.solver.assemble()

        # act
        self.solver.reset_b()

        # assert
        self.assertEqual(self.solver.b.getValues(0), 0)
        self.assertEqual(self.solver.b.getValues(1), 0)
    
    def test_solve(self):
        # arrange
        self.solver.insert_into_A(0, 0, 1)
        self.solver.insert_into_A(0, 1, 1)
        self.solver.insert_into_A(1, 0, 2)
        self.solver.insert_into_A(1, 1, 2)
        self.solver.insert_into_b(0, 2)
        self.solver.insert_into_b(1, 4)

        x = PETSc.Vec().createSeq(2)

        # act
        self.solver.solve(x)

        # assert
        self.assertAlmostEqual(x.getValues(0), 1)
        self.assertAlmostEqual(x.getValues(1), 1)
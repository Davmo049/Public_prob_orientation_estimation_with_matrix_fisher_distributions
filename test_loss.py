import unittest
import loss
import numpy as np
import torch

class TestAngleError(unittest.TestCase):

    def test_valid_rotation_matrix_correct(self):
        angle = np.pi/6
        cos = np.cos(angle)
        sin = np.sin(angle)
        R = torch.tensor([
            [1.0, 0,    0],
            [0,   cos,  sin],
            [0,   -sin, cos]
        ], dtype=torch.float32).view(1,3,3)
        R_gt = torch.eye(3).view(1,3,3)
        self.assertAlmostEqual(loss.angle_error(R, R_gt).numpy()[0], angle*180/np.pi, places=3)

    def test_batch_matrices_correct(self):
        angle1 = np.pi/12
        angle2 = np.pi/3
        cos1 = np.cos(angle1)
        sin1 = np.sin(angle1)
        cos2 = np.cos(angle2)
        sin2 = np.sin(angle2)
        R = torch.tensor([[
            [1.0, 0,     0],
            [0,   cos1,  sin1],
            [0,   -sin1, cos1]
        ],
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1.0]
        ]], dtype=torch.float32)
        R_gt = torch.tensor([[
            [1.0, 0,    0],
            [0,   1,    0],
            [0,   0,    1]
        ],
        [
            [cos2,  sin2, 0],
            [-sin2, cos2, 0],
            [0,     0,    1.0]
        ]], dtype=torch.float32)
        self.assertAlmostEqual(loss.angle_error(R, R_gt).numpy()[0], angle1*180/np.pi, places=3)
        self.assertAlmostEqual(loss.angle_error(R, R_gt).numpy()[1], angle2*180/np.pi, places=3)


if __name__ == '__main__':
    unittest.main()

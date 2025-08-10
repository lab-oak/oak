import numpy as np

import ctypes


class BuildTrajectory:
    def __init__(self, size):
        self.size = size

        self.actions = np.zeros((size, 31), dtype=np.int64)
        self.policy = np.zeros((size, 31), dtype=np.float32)
        self.eval = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)

    def raw_pointers(self, i: int):
        def ptr(x, dtype):
            return x[i].ctypes.data_as(ctypes.POINTER(dtype))

        return (
            ptr(self.actions, ctypes.c_int64),
            ptr(self.policy, ctypes.c_float),
            ptr(self.eval, ctypes.c_float),
            ptr(self.score, ctypes.c_float),
        )

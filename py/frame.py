import numpy as np
import torch

import ctypes


class Frame:
    def __init__(self, size: int):
        self.size = size

        self.m = torch.zeros((size, 1), dtype=np.uint8)
        self.n = torch.zeros((size, 1), dtype=np.uint8)

        self.battle = torch.zeros((size, 384), dtype=np.uint8)
        self.durations = torch.zeros((size, 8), dtype=np.uint8)
        self.result = torch.zeros((size, 1), dtype=np.uint8)

        self.p1_choices = torch.zeros((size, 9), dtype=np.uint8)
        self.p2_choices = torch.zeros((size, 9), dtype=np.uint8)

        self.p1_empirical = torch.zeros((size, 9), dtype=np.float32)
        self.p1_nash = torch.zeros((size, 9), dtype=np.float32)
        self.p2_empirical = torch.zeros((size, 9), dtype=np.float32)
        self.p2_nash = torch.zeros((size, 9), dtype=np.float32)

        self.empirical_value = torch.zeros((size, 1), dtype=np.float32)
        self.nash_value = torch.zeros((size, 1), dtype=np.float32)
        self.score = torch.zeros((size, 1), dtype=np.float32)

    def raw_pointers(self, i: int):
        return (
            ctypes.cast(self.m[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.n[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.battle[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.durations[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.result[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.p1_choices[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.p2_choices[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(
                self.p1_empirical[i].data_ptr(), ctypes.POINTER(ctypes.c_float)
            ),
            ctypes.cast(self.p1_nash[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(
                self.p2_empirical[i].data_ptr(), ctypes.POINTER(ctypes.c_float)
            ),
            ctypes.cast(self.p2_nash[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(
                self.empirical_value[i].data_ptr(), ctypes.POINTER(ctypes.c_float)
            ),
            ctypes.cast(self.nash_value[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.score[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
        )

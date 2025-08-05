import numpy as np
import ctypes


class Frame:
    def __init__(self, size: int):
        self.size = size

        self.m = np.zeros((size, 1), dtype=np.uint8)
        self.n = np.zeros((size, 1), dtype=np.uint8)

        self.battle = np.zeros((size, 384), dtype=np.uint8)
        self.durations = np.zeros((size, 8), dtype=np.uint8)
        self.result = np.zeros((size, 1), dtype=np.uint8)

        self.p1_choices = np.zeros((size, 9), dtype=np.uint8)
        self.p2_choices = np.zeros((size, 9), dtype=np.uint8)

        self.p1_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p1_nash = np.zeros((size, 9), dtype=np.float32)
        self.p2_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p2_nash = np.zeros((size, 9), dtype=np.float32)

        self.empirical_value = np.zeros((size, 1), dtype=np.float32)
        self.nash_value = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)

    def raw_pointers(self, i: int):
        return (
            self.m[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.n[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.battle[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.durations[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.result[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.p1_choices[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.p2_choices[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.p1_empirical[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.p1_nash[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.p2_empirical[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.p2_nash[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.empirical_value[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.nash_value[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.score[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

import ctypes
import array
import sys

import numpy as np

lib = ctypes.CDLL('./build/libtrain.so')

pokemon_in_dim = ctypes.c_int.in_dll(lib, "pokemon_in_dim").value
active_in_dim = ctypes.c_int.in_dll(lib, "active_in_dim").value

lib.read_battle_offsets.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_uint64,
]
lib.read_buffer_to_frames.restype = ctypes.c_int

lib.read_buffer_to_frames.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_uint8),  # battle
    ctypes.POINTER(ctypes.c_uint8),  # durations
    ctypes.POINTER(ctypes.c_uint8),  # result
    ctypes.POINTER(ctypes.c_uint8),  # p1_choices
    ctypes.POINTER(ctypes.c_uint8),  # p2_choices
    ctypes.POINTER(ctypes.c_float),  # p1_empirical
    ctypes.POINTER(ctypes.c_float),  # p1_nash
    ctypes.POINTER(ctypes.c_float),  # p2_empirical
    ctypes.POINTER(ctypes.c_float),  # p2_nash
    ctypes.POINTER(ctypes.c_float),  # empirical_value
    ctypes.POINTER(ctypes.c_float),  # nash_value
    ctypes.POINTER(ctypes.c_float),  # score
]
lib.read_buffer_to_frames.restype = ctypes.c_int

lib.encode_buffer.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_uint16), # p1_choice_indices
    ctypes.POINTER(ctypes.c_uint16), # p2_choice_indices
    ctypes.POINTER(ctypes.c_float),  # pokemon
    ctypes.POINTER(ctypes.c_float),  # active
    ctypes.POINTER(ctypes.c_float),  # hp
    ctypes.POINTER(ctypes.c_float),  # p1_empirical
    ctypes.POINTER(ctypes.c_float),  # p1_nash
    ctypes.POINTER(ctypes.c_float),  # p2_empirical
    ctypes.POINTER(ctypes.c_float),  # p2_nash
    ctypes.POINTER(ctypes.c_float),  # empirical_value
    ctypes.POINTER(ctypes.c_float),  # nash_value
    ctypes.POINTER(ctypes.c_float),  # score
]
lib.encode_buffer.restype = ctypes.c_int

# Python

def ptr(array: np.ndarray, dtype):
    return ctypes.cast(array.ctypes.data, ctypes.POINTER(dtype))

class FrameInput:
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
        self.p1_nash      = np.zeros((size, 9), dtype=np.float32)
        self.p2_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p2_nash      = np.zeros((size, 9), dtype=np.float32)

        self.empirical_value = np.zeros((size, 1), dtype=np.float32)
        self.nash_value      = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)

    def ptrs_for_index(self, i: int):
        return (
            self.m[i].ctypes.data,
            self.n[i].ctypes.data,
            self.battle[i].ctypes.data,
            self.durations[i].ctypes.data,
            self.result[i].ctypes.data,
            self.p1_choices[i].ctypes.data,
            self.p2_choices[i].ctypes.data,
            self.p1_empirical[i].ctypes.data,
            self.p1_nash[i].ctypes.data,
            self.p2_empirical[i].ctypes.data,
            self.p2_nash[i].ctypes.data,
            self.empirical_value[i].ctypes.data,
            self.nash_value[i].ctypes.data,
            self.score[i].ctypes.data,
        )


class EncodedFrameInput:
    def __init__(self, size):
        self.size = size

        self.m = np.zeros((size, 1), dtype=np.uint8)
        self.n = np.zeros((size, 1), dtype=np.uint8)

        self.p1_choice_indices = np.zeros((size, 9), dtype=np.uint16)
        self.p2_choice_indices = np.zeros((size, 9), dtype=np.uint16)

        self.pokemon = np.zeros((size, 2, 5, pokemon_in_dim), dtype=np.float32)
        self.active = np.zeros((size, 2, 1, active_in_dim), dtype=np.float32)
        self.hp = np.zeros((size, 2, 6), dtype=np.float32)

        self.p1_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p1_nash      = np.zeros((size, 9), dtype=np.float32)
        self.p2_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p2_nash      = np.zeros((size, 9), dtype=np.float32)

        self.empirical_value = np.zeros((size, 1), dtype=np.float32)
        self.nash_value      = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)

    def ptrs_for_index(self, i: int):
        return (
            ptr(self.m[i], ctypes.c_uint8),
            ptr(self.n[i], ctypes.c_uint8),
            ptr(self.p1_choice_indices[i], ctypes.c_uint16),
            ptr(self.p2_choice_indices[i], ctypes.c_uint16),
            ptr(self.pokemon[i], ctypes.c_float),
            ptr(self.active[i], ctypes.c_float),
            ptr(self.hp[i], ctypes.c_float),
            ptr(self.p1_empirical[i], ctypes.c_float),
            ptr(self.p1_nash[i], ctypes.c_float),
            ptr(self.p2_empirical[i], ctypes.c_float),
            ptr(self.p2_nash[i], ctypes.c_float),
            ptr(self.empirical_value[i], ctypes.c_float),
            ptr(self.nash_value[i], ctypes.c_float),
            ptr(self.score[i], ctypes.c_float),
        )


def read_battle_offsets(path, n):
    path_bytes = path.encode('utf-8')
    offsets = np.zeros(n, dtype=np.uint16)

    args = (
        ctypes.c_char_p(path_bytes),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_uint64(n),
    )
    res = lib.read_battle_offsets(*args)

def read_buffer_to_frames(path, max_frames, frame_input : FrameInput):

    path_bytes = path.encode('utf-8')

    count = min(max_frames, frame_input.size)

    args = (ctypes.c_char_p(path_bytes), ctypes.c_uint64(count)) + tuple(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 if j < 7 else ctypes.c_float))
        for j, ptr in enumerate(frame_input.ptrs_for_index(0))
    )
    res = lib.read_buffer_to_frames(*args)

    return res

def encode_buffer(path, max_frames, encoded_frame_input):

    path_bytes = path.encode('utf-8')

    count = min(max_frames, encoded_frame_input.size)

    args = (ctypes.c_char_p(path_bytes), ctypes.c_uint64(count)) + encoded_frame_input.ptrs_for_index(0)
        
    res = lib.encode_buffer(*args)

    return res
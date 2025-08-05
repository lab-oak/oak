from frame import Frame

import ctypes
import os

import numpy as np

lib = ctypes.CDLL("./build/libpyoak.so")

lib.get_compressed_battles_helper.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
]
lib.get_compressed_battles_helper.restype = ctypes.c_int

lib.uncompress_training_frames.argtypes = [
    ctypes.c_char_p,
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

# Python

# Returns a list tuples containing: a bytes object, the number of wrtten to the bytes
def read_battle_data(path: str, max_battles=1_000_000) -> list[tuple[bytes, int]]:
    path_bytes = path.encode("utf-8")

    offsets = (ctypes.c_uint16 * max_battles)()
    frame_counts = (ctypes.c_uint16 * max_battles)()
    buffer_size = os.path.getsize(path)
    buffer = (ctypes.c_char * int(buffer_size))()

    args = (
        ctypes.c_char_p(path_bytes),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
        ctypes.cast(offsets, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(frame_counts, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_uint64(max_battles),
    )

    n = lib.get_compressed_battles_helper(*args)

    if n < 0:
        raise RuntimeError("Failed to read battle data")

    offset = 0
    result = []
    for i in range(n):
        sz = offsets[i]
        result.append((bytes(buffer[offset : offset + sz]), int(frame_counts[i])))
        offset += sz

    return result

# convert bytes object into Frames
def get_frames(data: bytes, frame_count: int) -> Frame:
    frames = Frame(frame_count)
    args = (ctypes.c_char_p(data),) + frames.raw_pointers(0)
    lib.uncompress_training_frames(*args)
    return frames

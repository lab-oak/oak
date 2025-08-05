from frame import Frame

import ctypes
import os

import numpy as np

lib = ctypes.CDLL("./build/libpyoak.so")

# functions
lib.read_battle_offsets.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_uint64,
]
lib.read_battle_offsets.restype = ctypes.c_int

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

lib.read_buffer_to_frames.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_float,
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
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_int64),  # p1_choice_indices
    ctypes.POINTER(ctypes.c_int64),  # p2_choice_indices
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

lib.encode_buffer_multithread.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_int64),  # p1_choice_indices
    ctypes.POINTER(ctypes.c_int64),  # p2_choice_indices
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
lib.encode_buffer_multithread.restype = ctypes.c_uint64

# Python

import numpy as np
import ctypes


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
    print(frame_count)
    frames = Frame(frame_count)
    args = (ctypes.c_char_p(data),) + frames.raw_pointers(0)
    lib.uncompress_training_frames(*args)
    return frames


def read_battle_offsets(path, n):
    path_bytes = path.encode("utf-8")
    offsets = np.zeros(n, dtype=np.uint16)

    args = (
        ctypes.c_char_p(path_bytes),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_uint64(n),
    )
    res = lib.read_battle_offsets(*args)
    return offsets, res


def read_buffer_to_frames(
    path: str,
    max_frames: int,
    frame_input: Frame,
    start_index: int = 0,
    write_prob: float = 1,
):
    path_bytes = path.encode("utf-8")
    max_count = min(max_frames, frame_input.size)
    args = (
        ctypes.c_char_p(path_bytes),
        ctypes.c_uint64(max_count),
        ctypes.c_float(write_prob),
    ) + frame_input.raw_pointers(start_index)
    count = lib.read_buffer_to_frames(*args)
    return count


def encode_buffer(
    path: str,
    max_frames: int,
    encoded_frame_input: EncodedFrame,
    start_index: int = 0,
    write_prob: float = 1,
):
    path_bytes = path.encode("utf-8")
    max_count = min(max_frames, encoded_frame_input.size)
    args = (
        ctypes.c_char_p(path_bytes),
        ctypes.c_uint64(max_count),
        ctypes.c_float(write_prob),
    ) + encoded_frame_input.raw_pointers(start_index)
    count = lib.encode_buffer(*args)
    return count


def encode_buffers(
    paths: list[str],
    threads: int,
    max_count: int,
    encoded_frame_input: EncodedFrame,
    start_index: int = 0,
    write_prob: float = 1,
):
    encoded_paths = [path.encode("utf-8") for path in paths]
    arr = (ctypes.c_char_p * len(encoded_paths))(*encoded_paths)
    max_count = min(max_count, encoded_frame_input.size)
    args = (
        arr,
        ctypes.c_uint64(len(encoded_paths)),
        ctypes.c_uint64(threads),
        ctypes.c_uint64(max_count),
        ctypes.c_float(write_prob),
    ) + encoded_frame_input.raw_pointers(start_index)
    count = lib.encode_buffer_multithread(*args)
    return count

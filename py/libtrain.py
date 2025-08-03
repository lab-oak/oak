from frame import Frame

import ctypes

import torch

import numpy as np

lib = ctypes.CDLL("./build/libtrain.so")

# net hyperparams
pokemon_in_dim = ctypes.c_int.in_dll(lib, "pokemon_in_dim").value
active_in_dim = ctypes.c_int.in_dll(lib, "active_in_dim").value
pokemon_hidden_dim = ctypes.c_int.in_dll(lib, "pokemon_hidden_dim").value
pokemon_out_dim = ctypes.c_int.in_dll(lib, "pokemon_out_dim").value
active_hidden_dim = ctypes.c_int.in_dll(lib, "active_hidden_dim").value
active_out_dim = ctypes.c_int.in_dll(lib, "active_out_dim").value
side_out_dim = ctypes.c_int.in_dll(lib, "side_out_dim").value
hidden_dim = ctypes.c_int.in_dll(lib, "hidden_dim").value
value_hidden_dim = ctypes.c_int.in_dll(lib, "value_hidden_dim").value
policy_hidden_dim = ctypes.c_int.in_dll(lib, "policy_hidden_dim").value
policy_out_dim = ctypes.c_int.in_dll(lib, "policy_out_dim").value

# dimension labels for encoding, lists of strings
pokemon_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(
    lib, "pokemon_dim_labels"
)
active_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "active_dim_labels")
policy_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "policy_dim_labels")


def char_pp_to_str_list(char_pp, count):
    return [char_pp[i].decode("utf-8") for i in range(count)]


pokemon_dim_labels: list[str] = char_pp_to_str_list(
    pokemon_dim_labels_raw, pokemon_in_dim
)
active_dim_labels: list[str] = char_pp_to_str_list(active_dim_labels_raw, active_in_dim)
policy_dim_labels: list[str] = char_pp_to_str_list(
    policy_dim_labels_raw, policy_out_dim
)

# functions
lib.read_battle_offsets.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_uint64,
]
lib.read_buffer_to_frames.restype = ctypes.c_int

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


class EncodedFrame:
    def __init__(self, size):
        self.size = size

        self.m = torch.zeros((size, 1), dtype=torch.uint8)
        self.n = torch.zeros((size, 1), dtype=torch.uint8)

        self.p1_choice_indices = torch.zeros((size, 9), dtype=torch.int64)
        self.p2_choice_indices = torch.zeros((size, 9), dtype=torch.int64)

        self.pokemon = torch.zeros((size, 2, 5, pokemon_in_dim), dtype=torch.float32)
        self.active = torch.zeros((size, 2, 1, active_in_dim), dtype=torch.float32)
        self.hp = torch.zeros((size, 2, 6, 1), dtype=torch.float32)

        self.p1_empirical = torch.zeros((size, 9), dtype=torch.float32)
        self.p1_nash = torch.zeros((size, 9), dtype=torch.float32)
        self.p2_empirical = torch.zeros((size, 9), dtype=torch.float32)
        self.p2_nash = torch.zeros((size, 9), dtype=torch.float32)

        self.empirical_value = torch.zeros((size, 1), dtype=torch.float32)
        self.nash_value = torch.zeros((size, 1), dtype=torch.float32)
        self.score = torch.zeros((size, 1), dtype=torch.float32)

    def clear(self):
        self.m.detach_().zero_()
        self.n.detach_().zero_()
        self.p1_choice_indices.detach_().zero_()
        self.p2_choice_indices.detach_().zero_()

        self.pokemon.detach_().zero_()
        self.active.detach_().zero_()
        self.hp.detach_().zero_()

        self.p1_empirical.detach_().zero_()
        self.p1_nash.detach_().zero_()
        self.p2_empirical.detach_().zero_()
        self.p2_nash.detach_().zero_()

        self.empirical_value.detach_().zero_()
        self.nash_value.detach_().zero_()
        self.score.detach_().zero_()

    def raw_pointers(self, i: int):
        return (
            ctypes.cast(self.m[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(self.n[i].data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(
                self.p1_choice_indices[i].data_ptr(), ctypes.POINTER(ctypes.c_int64)
            ),
            ctypes.cast(
                self.p2_choice_indices[i].data_ptr(), ctypes.POINTER(ctypes.c_int64)
            ),
            ctypes.cast(self.pokemon[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.active[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.hp[i].data_ptr(), ctypes.POINTER(ctypes.c_float)),
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


def read_battle_offsets(path, n):
    path_bytes = path.encode("utf-8")
    offsets = np.zeros(n, dtype=np.uint16)

    args = (
        ctypes.c_char_p(path_bytes),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_uint64(n),
    )
    res = lib.read_battle_offsets(*args)
    return res


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

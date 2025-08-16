from frame import Frame
from build_trajectory import BuildTrajectory

import ctypes
import os

import numpy as np

lib = ctypes.CDLL("./build/libpyoak.so")

# battle net hyperparams
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

# build net hyperparams
# TODO

species_move_list_size = ctypes.c_int.in_dll(lib, "species_move_list_size").value
species_move_list_raw = ctypes.POINTER(ctypes.c_int).in_dll(
    lib, "species_move_list_ptrs"
)

species_move_list = []
for _ in range(species_move_list_size):
    species_move_list.append(
        (
            int(species_move_list_raw[2 * _]),
            int(species_move_list_raw[2 * _ + 1]),
        )
    )

# dimension labels for encoding, lists of strings
species_names_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "species_names")
move_names_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "move_names")
pokemon_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(
    lib, "pokemon_dim_labels"
)
active_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "active_dim_labels")
policy_dim_labels_raw = ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "policy_dim_labels")


def char_pp_to_str_list(char_pp, count):
    return [char_pp[i].decode("utf-8") for i in range(count)]


move_names: list[str] = char_pp_to_str_list(move_names_raw, 166)
species_names: list[str] = char_pp_to_str_list(species_names_raw, 152)
pokemon_dim_labels: list[str] = char_pp_to_str_list(
    pokemon_dim_labels_raw, pokemon_in_dim
)
active_dim_labels: list[str] = char_pp_to_str_list(active_dim_labels_raw, active_in_dim)
policy_dim_labels: list[str] = char_pp_to_str_list(
    policy_dim_labels_raw, policy_out_dim
)
policy_dim_labels.append("")

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

lib.uncompress_and_encode_training_frames.argtypes = [
    ctypes.c_char_p,
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

lib.read_build_trajectories.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.read_build_trajectories.restype = ctypes.c_int

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


class EncodedFrame:
    def __init__(self, size):
        self.size = size

        self.m = np.zeros((size, 1), dtype=np.uint8)
        self.n = np.zeros((size, 1), dtype=np.uint8)

        self.p1_choice_indices = np.zeros((size, 9), dtype=np.int64)
        self.p2_choice_indices = np.zeros((size, 9), dtype=np.int64)

        self.pokemon = np.zeros((size, 2, 5, pokemon_in_dim), dtype=np.float32)
        self.active = np.zeros((size, 2, 1, active_in_dim), dtype=np.float32)
        self.hp = np.zeros((size, 2, 6, 1), dtype=np.float32)

        self.p1_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p1_nash = np.zeros((size, 9), dtype=np.float32)
        self.p2_empirical = np.zeros((size, 9), dtype=np.float32)
        self.p2_nash = np.zeros((size, 9), dtype=np.float32)

        self.empirical_value = np.zeros((size, 1), dtype=np.float32)
        self.nash_value = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)

    def clear(self):
        self.m.fill(0)
        self.n.fill(0)
        self.p1_choice_indices.fill(0)
        self.p2_choice_indices.fill(0)

        self.pokemon.fill(0)
        self.active.fill(0)
        self.hp.fill(0)

        self.p1_empirical.fill(0)
        self.p1_nash.fill(0)
        self.p2_empirical.fill(0)
        self.p2_nash.fill(0)

        self.empirical_value.fill(0)
        self.nash_value.fill(0)
        self.score.fill(0)

    def raw_pointers(self, i: int):
        def ptr(x, dtype):
            return x[i].ctypes.data_as(ctypes.POINTER(dtype))

        return (
            ptr(self.m, ctypes.c_uint8),
            ptr(self.n, ctypes.c_uint8),
            ptr(self.p1_choice_indices, ctypes.c_int64),
            ptr(self.p2_choice_indices, ctypes.c_int64),
            ptr(self.pokemon, ctypes.c_float),
            ptr(self.active, ctypes.c_float),
            ptr(self.hp, ctypes.c_float),
            ptr(self.p1_empirical, ctypes.c_float),
            ptr(self.p1_nash, ctypes.c_float),
            ptr(self.p2_empirical, ctypes.c_float),
            ptr(self.p2_nash, ctypes.c_float),
            ptr(self.empirical_value, ctypes.c_float),
            ptr(self.nash_value, ctypes.c_float),
            ptr(self.score, ctypes.c_float),
        )


# convert bytes object into Frames
def get_encoded_frames(data: bytes, frame_count: int) -> EncodedFrame:
    encoded_frames = EncodedFrame(frame_count)
    args = (ctypes.c_char_p(data),) + encoded_frames.raw_pointers(0)
    lib.uncompress_and_encode_training_frames(*args)
    return encoded_frames


# convert bytes object into Frames
def read_build_trajectories(path) -> BuildTrajectory:
    buffer_size = int(os.path.getsize(path) / 128)
    path_bytes = path.encode("utf-8")

    trajectories = BuildTrajectory(buffer_size)
    args = (ctypes.c_char_p(path_bytes),) + trajectories.raw_pointers(0)
    count = lib.read_build_trajectories(*args)
    return trajectories


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

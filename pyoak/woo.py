import ctypes
import os
import sys

import numpy as np

from typing import List

if sys.platform == "win32":
    libname = "libpyoak.dll"
else:
    libname = "libpyoak.so"
lib = ctypes.CDLL(os.path.join("release", libname))

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
build_policy_hidden_dim = ctypes.c_int.in_dll(lib, "build_policy_hidden_dim").value
build_value_hidden_dim = ctypes.c_int.in_dll(lib, "build_value_hidden_dim").value
build_max_actions = ctypes.c_int.in_dll(lib, "build_max_actions").value

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

frame_target_types = [
    ctypes.POINTER(ctypes.c_uint32),  # iterations
    ctypes.POINTER(ctypes.c_float),  # p1_empirical
    ctypes.POINTER(ctypes.c_float),  # p1_nash
    ctypes.POINTER(ctypes.c_float),  # p2_empirical
    ctypes.POINTER(ctypes.c_float),  # p2_nash
    ctypes.POINTER(ctypes.c_float),  # empirical_value
    ctypes.POINTER(ctypes.c_float),  # nash_value
    ctypes.POINTER(ctypes.c_float),  # score
]

encoded_frame_input_types = [
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_int64),  # p1_choice_indices
    ctypes.POINTER(ctypes.c_int64),  # p2_choice_indices
    ctypes.POINTER(ctypes.c_float),  # pokemon
    ctypes.POINTER(ctypes.c_float),  # active
    ctypes.POINTER(ctypes.c_float),  # hp
]

lib.index_compressed_battle_frames.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
]
lib.index_compressed_battle_frames.restype = ctypes.c_int

lib.test_consistency.argtypes = [
    ctypes.c_uint64,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
]
lib.test_consistency.restype = ctypes.c_int

lib.uncompress_training_frames.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_uint8),  # battle
    ctypes.POINTER(ctypes.c_uint8),  # durations
    ctypes.POINTER(ctypes.c_uint8),  # result
    ctypes.POINTER(ctypes.c_uint8),  # p1_choices
    ctypes.POINTER(ctypes.c_uint8),  # p2_choices
] + frame_target_types

lib.uncompress_and_encode_training_frames.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint8),  # m
    ctypes.POINTER(ctypes.c_uint8),  # n
    ctypes.POINTER(ctypes.c_int64),  # p1_choice_indices
    ctypes.POINTER(ctypes.c_int64),  # p2_choice_indices
    ctypes.POINTER(ctypes.c_float),  # pokemon
    ctypes.POINTER(ctypes.c_float),  # active
    ctypes.POINTER(ctypes.c_float),  # hp
] + frame_target_types

lib.read_build_trajectories.argtypes = [
    ctypes.c_uint64,  # max_count
    ctypes.c_uint64,  # threads
    ctypes.c_uint64,  # n_paths
    ctypes.POINTER(ctypes.c_char_p),  # paths
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]
lib.read_build_trajectories.restype = ctypes.c_uint64

lib.sample_from_battle_data_files.argtypes = (
    [
        ctypes.c_uint64,  # max_count
        ctypes.c_uint64,  # threads
        ctypes.c_uint64,  # max_battle_length
        ctypes.c_uint64,  # min_iterations
        ctypes.c_uint64,  # n_paths
        ctypes.POINTER(ctypes.c_char_p),  # paths
        ctypes.POINTER(ctypes.c_int32),  # n_battles
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),  # offsets
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),  # n_frames
    ]
    + encoded_frame_input_types
    + frame_target_types
)
lib.sample_from_battle_data_files.restype = ctypes.c_uint64

lib.print_battle_data.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # battle
    ctypes.POINTER(ctypes.c_uint8),  # durations
]


def test_consistency(
    max_games: int, network_path: str, data_path: str, max_battles=1_000_000
) -> list[tuple[bytes, int]]:
    network_path_bytes = network_path.encode("utf-8")
    path_bytes = data_path.encode("utf-8")

    offsets = (ctypes.c_uint16 * max_battles)()
    frame_counts = (ctypes.c_uint16 * max_battles)()
    buffer_size = os.path.getsize(data_path)
    buffer = (ctypes.c_char * int(buffer_size))()

    args = (
        ctypes.c_uint64(max_games),
        ctypes.c_char_p(network_path_bytes),
        ctypes.c_char_p(path_bytes),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
        ctypes.cast(offsets, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(frame_counts, ctypes.POINTER(ctypes.c_uint16)),
    )

    n = lib.test_consistency(*args)

    return n


def print_battle_data(frames, i):
    args = frames.raw_pointers(i)
    lib.print_battle_data(args[2], args[3])


class BuildTrajectories:
    def __init__(self, size):
        self.size = size

        self.actions = np.zeros((size, 31, 1), dtype=np.int64)
        self.mask = np.zeros((size, 31, build_max_actions), dtype=np.int64)
        self.policy = np.zeros((size, 31, 1), dtype=np.float32)
        self.value = np.zeros((size, 1), dtype=np.float32)
        self.score = np.zeros((size, 1), dtype=np.float32)
        self.start = np.zeros((size, 1), dtype=np.int64)
        self.end = np.zeros((size, 1), dtype=np.int64)

    def raw_pointers(self, i: int):
        def ptr(x, dtype):
            return x[i].ctypes.data_as(ctypes.POINTER(dtype))

        return (
            ptr(self.actions, ctypes.c_int64),
            ptr(self.mask, ctypes.c_int64),
            ptr(self.policy, ctypes.c_float),
            ptr(self.value, ctypes.c_float),
            ptr(self.score, ctypes.c_float),
            ptr(self.start, ctypes.c_int64),
            ptr(self.end, ctypes.c_int64),
        )


# convert bytes object into BattleFrames
def read_build_trajectories(
    paths: List[str], buffer_size: int, threads: int
) -> [BuildTrajectories, int]:
    encoded_paths: List[str] = [p.encode("utf-8") for p in paths]
    trajectories = BuildTrajectories(buffer_size)
    args = (
        ctypes.c_uint64(buffer_size),
        ctypes.c_uint64(threads),
        ctypes.c_uint64(len(paths)),
        (ctypes.c_char_p * len(paths))(*encoded_paths),
    ) + trajectories.raw_pointers(0)
    count = lib.read_build_trajectories(*args)
    return trajectories, count


def sample_from_battle_data_files(
    indexer: SampleIndexer,
    encoded_frames: EncodedBattleFrames,
    threads: int,
    max_game_length: int = 10000,
    minimum_iterations: int = 1,
):

    paths: List[str] = []
    n_battles: List[int] = []
    offsets: List[List[int]] = []
    frames: List[List[int]] = []

    for p, offset_frames_list in indexer.data.items():
        paths.append(p.encode("utf-8"))
        n_battles.append(len(offset_frames_list))
        offsets.append([])
        frames.append([])
        for o, f in offset_frames_list:
            offsets[-1].append(o)
            frames[-1].append(f)

    def lists_to_int_pp(lists: list[list[int]]):
        inner_arrays = [(ctypes.c_int32 * len(inner))(*inner) for inner in lists]
        outer_array = (ctypes.POINTER(ctypes.c_int32) * len(inner_arrays))(
            *[ctypes.cast(arr, ctypes.POINTER(ctypes.c_int32)) for arr in inner_arrays]
        )

        return inner_arrays, outer_array

    offset_inner, offsets_pp = lists_to_int_pp(offsets)
    frame_inner, frames_pp = lists_to_int_pp(frames)

    args = (
        ctypes.c_uint64(encoded_frames.size),
        ctypes.c_uint64(threads),
        ctypes.c_uint64(max_game_length),
        ctypes.c_uint64(minimum_iterations),
        ctypes.c_uint64(len(indexer.data)),
        (ctypes.c_char_p * len(paths))(*paths),
        (ctypes.c_int32 * len(n_battles))(*n_battles),
        offsets_pp,
        frames_pp,
    ) + encoded_frames.raw_pointers(0)

    count = lib.sample_from_battle_data_files(*args)
    return count


# Get all files in all subdirs or root_dir with the specificed extension.
# The files are sorted newest first
def find_data_files(root_dir, ext):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                full_path = os.path.join(dirpath, filename)
                files.append(full_path)
    files.sort(key=os.path.getctime, reverse=True)
    return files


def save_args(namespace, path):
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "args")

    with open(out_path, "w") as f:
        for key, value in vars(namespace).items():
            f.write(f"--{key}={value}\n")

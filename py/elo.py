import os
import struct
import torch
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import subprocess

import py_oak
import torch_oak

parser = argparse.ArgumentParser(description="Parameter testing for battle agents.")

parser.add_argument("--working-dir", default=None, type=str)
parser.add_argument("--vs-path", default="./release/vs", type=str)
parser.add_argument("--search-iterations", default=2**12, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--max-agents", default=32, type=int)
parser.add_argument("--n-delete", default=8, type=int)
parser.add_argument("--games-per-update", default=2**10, type=int)

args = parser.parse_args()


class ID:
    def __init__(self, net_hash, bandit_name, policy_mode):
        self.net_hash = net_hash
        assert len(bandit_name) < 15, f"Error: bandit-name too long: {bandit_name}"
        self.bandit_name = bandit_name
        assert len(policy_mode) == 1, f"Error: policy mode must be char: {policy_mode}"
        self.policy_mode = policy_mode

    def write(self, f):
        f.write(struct.pack("<Q", self.net_hash))
        bn = self.bandit_name.encode("utf8")
        bn = bn.ljust(15, b"\0")
        f.write(bn)
        f.write(self.policy_mode.encode("utf8"))


class Global:
    def __init__(self):
        self.ratings: Dict[ID, float] = {}
        self.ucb: Dict[ID, tuple[float, int]] = {}
        self.results: Dict[tuple[ID, ID], tuple[int, int, int]] = {}
        self.directory: Dict[int, str] = {}

    def remove_id(self, target_id: ID):
        self.ratings = {k: v for k, v in self.ratings.items() if k != target_id}
        self.ucb = {k: v for k, v in self.ucb.items() if k != target_id}
        self.results = {
            k: v
            for k, v in self.results.items()
            if k[0] != target_id and k[1] != target_id
        }
        if target_id.net_hash in self.directory:
            del self.directory[target_id.net_hash]

    def sample_ids(self):
        if not self.ucb:
            raise RuntimeError("No IDs to sample from")

        ids_sorted = sorted(self.ucb.items(), key=lambda x: x[1][0], reverse=True)
        if len(ids_sorted) < 2:
            raise RuntimeError("Need at least 2 IDs")

        id1 = ids_sorted[0][0]
        id2 = ids_sorted[1][0]
        return id1, id2


glob = Global()


def read_files():
    base = args.working_dir

    # RATINGS
    path = os.path.join(base, "ratings")
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                data = f.read(8 + 15 + 1 + 4)
                if len(data) < 8 + 15 + 1 + 4:
                    break
                net_hash = struct.unpack("<Q", data[:8])[0]
                bandit = data[8:23].rstrip(b"\0").decode("ascii")
                mode = data[23:24].decode("ascii")
                rating = struct.unpack("<f", data[24:28])[0]
                obj = ID(net_hash, bandit, mode)
                glob.ratings[obj] = rating
                glob.ucb[obj] = (0.0, 0)

    # RESULTS
    path = os.path.join(base, "results")
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                id_chunk = f.read((8 + 15 + 1) * 2 + 4 * 3)
                if len(id_chunk) < (8 + 15 + 1) * 2 + 12:
                    break

                p = 0

                def read_id():
                    nonlocal p
                    nh = struct.unpack("<Q", id_chunk[p : p + 8])[0]
                    p += 8
                    bn = id_chunk[p : p + 15].rstrip(b"\0").decode("ascii")
                    p += 15
                    pm = id_chunk[p : p + 1].decode("ascii")
                    p += 1
                    return ID(nh, bn, pm)

                id1 = read_id()
                id2 = read_id()
                w, l, d = struct.unpack("<III", id_chunk[p : p + 12])
                glob.results[(id1, id2)] = (w, l, d)

    # DIRECTORY
    path = os.path.join(base, "directory")
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                h_bytes = f.read(8)
                if len(h_bytes) < 8:
                    break
                net_hash = struct.unpack("<Q", h_bytes)[0]
                path_bytes = []
                while True:
                    c = f.read(1)
                    if not c or c == b"\0":
                        break
                    path_bytes.append(c)
                full_path = b"".join(path_bytes).decode("utf-8")
                glob.directory[net_hash] = full_path


def write_files():
    base = args.working_dir

    # RATINGS
    with open(os.path.join(base, "ratings"), "wb") as f:
        for obj, rating in glob.ratings.items():
            obj.write(f)
            f.write(struct.pack("<f", rating))

    # RESULTS
    with open(os.path.join(base, "results"), "wb") as f:
        for (id1, id2), (w, l, d) in glob.results.items():
            id1.write(f)
            id2.write(f)
            f.write(struct.pack("<III", w, l, d))

    # DIRECTORY
    with open(os.path.join(base, "directory"), "wb") as f:
        for nh, path_str in glob.directory.items():
            f.write(struct.pack("<Q", nh))
            f.write(path_str.encode("utf-8"))
            f.write(b"\0")

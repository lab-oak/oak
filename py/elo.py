import os
import struct
import torch
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import subprocess
import math

import py_oak
import torch_oak

parser = argparse.ArgumentParser(description="Parameter testing for battle agents.")

parser.add_argument("--working-dir", default=None, type=str)
parser.add_argument("--net-path", default=".", type=str)
parser.add_argument("--vs-path", default="./release/vs", type=str)
parser.add_argument("--search-iterations", default=2**12, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--max-agents", default=32, type=int)
parser.add_argument("--n-delete", default=8, type=int)
parser.add_argument("--games-per-save", default=2**10, type=int)
parser.add_argument("--saves-per-update", default=1, type=int)
parser.add_argument("--teams", default="", type=str)
parser.add_argument("--exp3-param-min", default=0.001, type=float)
parser.add_argument("--exp3-param-max", default=5.0, type=float)
parser.add_argument("--ucb-param-min", default=0.001, type=float)
parser.add_argument("--ucb-param-max", default=5.0, type=float)
parser.add_argument("--allow-policy", default=False, type=bool)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--c", default=1.414, type=float)
parser.add_argument("--K", default=8, type=float)

args = parser.parse_args()

# Util functions


def log_uniform(min, max):
    log_min = math.log(min)
    log_max = math.log(max)
    log_random = random.uniform(log_min, log_max)
    return math.exp(log_random)


class ID:
    def __init__(self, net_hash, bandit_name, policy_mode):
        self.net_hash = net_hash
        assert len(bandit_name) < 15, f"Error: bandit-name too long: {bandit_name}"
        self.bandit_name = bandit_name
        assert len(policy_mode) == 1, f"Error: policy mode must be char: {policy_mode}"
        self.policy_mode = policy_mode

    def __eq__(self, other):
        if not isinstance(other, ID):
            return NotImplemented
        return (
            self.net_hash == other.net_hash
            and self.bandit_name == other.bandit_name
            and self.policy_mode == other.policy_mode
        )

    def __hash__(self):
        return hash(
            (self.net_hash, self.bandit_name, self.policy_mode)
        )  # combine fields safely

    def write(self, f):
        f.write(struct.pack("<Q", self.net_hash))
        bn = self.bandit_name.encode("utf8")
        bn = bn.ljust(15, b"\0")
        f.write(bn)
        f.write(self.policy_mode.encode("utf8"))

    def print(self):
        print(self.net_hash, self.bandit_name, self.policy_mode)


def less_than(id1: ID, id2: ID):
    return (id1.net_hash, id1.bandit_name, id1.policy_mode) < (
        id2.net_hash,
        id2.bandit_name,
        id2.policy_mode,
    )


def permute(id: ID) -> ID:
    child = id


class Global:

    default_rating: int = 1200
    c: float = args.c

    def __init__(self):

        self.ratings: Dict[ID, float] = {}
        self.ucb: Dict[ID, tuple[float, int]] = {}
        self.results: Dict[tuple[ID, ID], List[int, int, int]] = {}
        self.directory: Dict[int, str] = {}

    def fill_from_path(self, path, n=args.max_agents):
        net_files = py_oak.find_data_files(path, ext=".net")

        for file in net_files:
            network = torch_oak.BattleNetwork()
            with open(file, "rb") as f:
                network.read_parameters(f)
            network_hash = network.hash()
            self.directory[network_hash] = file

        # assume 0 hash is not possible
        self.directory[0] = "mc"

        for _ in range(n):
            agent = self.new_agent()
            self.ratings[agent] = self.default_rating
            self.ucb[agent] = [0, 0]

    def new_agent(
        self,
    ):
        net_hash, net_path = random.choice(list(self.directory.items()))

        base_bandits = ["exp3-", "ucb-"]
        if args.allow_policy:
            base_bandits.append("pexp3-")
            base_bandits.append("pucb-")
        bandit = random.choice(base_bandits)

        param = None
        if bandit.startswith("pexp3") or bandit.startswith("exp3"):
            param = random.uniform(args.exp3_param_min, args.exp3_param_max)
        elif bandit.startswith("pucb") or bandit.startswith("ucb"):
            param = random.uniform(args.ucb_param_min, args.ucb_param_max)
        else:
            assert false, "bad bandit name"
        bandit_name = bandit + str(param)[:5]

        selection_modes = ["n", "e", "x"]
        selection_mode = random.choice(selection_modes)

        id = ID(net_hash, bandit_name, selection_mode)
        return id

    def remove_id(self, target_id: ID):
        return NotImplemented

    def select(self):

        N = sum(map(lambda kv: kv[1][1], self.ucb.items()))

        ids_sorted = sorted(
            self.ucb.items(),
            key=lambda kv: (kv[1][0]) + (self.c * math.sqrt(N) / (kv[1][1] + 1)),
            reverse=True,
        )
        if len(ids_sorted) < 2:
            raise RuntimeError("Need at least 2 IDs")

        first, second = ids_sorted[:2]

        # increment visits first (virtual loss)
        first[1][1] = first[1][1] + 1
        second[1][1] = second[1][1] + 1

        id1 = first[0]
        id2 = second[0]
        if less_than(id2, id1):
            id1, id2 = id2, id1
        return id1, id2

    def update(self, lesserID, greaterID, data):
        score = (data[0] + 0.5 * data[1]) / 2

        # Update results
        wdl = self.results.get((lesserID, greaterID), [0, 0, 0])
        wdl[0] += data[0]
        wdl[1] += data[1]
        wdl[2] += data[2]
        self.results[(lesserID, greaterID)] = wdl

        # Elo update
        R_lesser = self.ratings[lesserID]
        R_greater = self.ratings[greaterID]

        E_lesser = 1 / (1 + 10 ** ((R_greater - R_lesser) / 400))
        E_greater = 1 - E_lesser

        self.ratings[lesserID] += args.K * (score - E_lesser)
        self.ratings[greaterID] += args.K * ((1 - score) - E_greater)

        # ucb, update value only
        v, n = self.ucb[lesserID]
        total_v = v * (n - 1)
        assert n > 0, "Bad UCB update"
        self.ucb[lesserID] = [(total_v + score) / n, n]
        v, n = self.ucb[greaterID]
        assert n > 0, "Bad UCB update"
        total_v = v * (n - 1)
        self.ucb[greaterID] = [(total_v + (1 - score)) / n, n]

    def remove_lowest(self, n):

        sorted_ratings = sorted(self.ucb.items(), key=lambda kv: kv[1])
        for kv in sorted_ratings[:n]:
            del self.ucb[kv[0]]

    def reset_visits(self):
        for key, value in self.ucb:
            value[1] = 0


glob = Global()


def read_files():

    print("Reading files")

    base = args.working_dir

    with open(os.path.join(base, "ratings"), "rb") as f:
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

    with open(os.path.join(base, "ucb"), "rb") as f:

        temp = dict()

        while True:
            data = f.read(8 + 15 + 1 + 4 + 4)
            if len(data) < 8 + 15 + 1 + 4 + 4:
                break
            net_hash = struct.unpack("<Q", data[:8])[0]
            bandit = data[8:23].rstrip(b"\0").decode("ascii")
            mode = data[23:24].decode("ascii")
            score, visits = struct.unpack("<fI", data[24:32])
            obj = ID(net_hash, bandit, mode)
            temp[obj] = [score, visits]

        sorted_ratings = sorted(temp.items(), key=lambda kv: kv[1], reverse=True)
        for key, value in sorted_ratings[:args.max_agents]:
            # value[1] = 0
            glob.ucb[key] = value            

    with open(os.path.join(base, "results"), "rb") as f:
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
            glob.results[(id1, id2)] = [w, l, d]

    with open(os.path.join(base, "directory"), "rb") as f:
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

    with open(os.path.join(base, "ratings"), "wb") as f:
        for obj, rating in glob.ratings.items():
            obj.write(f)
            f.write(struct.pack("<f", rating))

    with open(os.path.join(base, "ucb"), "wb") as f:
        for obj, data in glob.ucb.items():
            obj.write(f)
            f.write(struct.pack("<fI", data[0], data[1]))

    with open(os.path.join(base, "results"), "wb") as f:
        for (id1, id2), (w, l, d) in glob.results.items():
            id1.write(f)
            id2.write(f)
            f.write(struct.pack("<III", w, l, d))

    with open(os.path.join(base, "directory"), "wb") as f:
        for nh, path_str in glob.directory.items():
            f.write(struct.pack("<Q", nh))
            f.write(path_str.encode("utf-8"))
            f.write(b"\0")


def setup():

    if args.working_dir is None:
        import datetime

        now = datetime.datetime.now()
        working_dir = now.strftime("elo-%Y-%m-%d-%H:%M:%S")
        os.makedirs(working_dir, exist_ok=False)
        args.working_dir = working_dir

        glob.fill_from_path(args.net_path)

    else:
        read_files()


def run_once() -> [ID, ID, [int, int, int]]:

    lesserID, greaterID = glob.select()

    cmd = [
        args.vs_path,
        "--threads=1",
        "--max-games=1",
        "--skip-save",
        f"--teams={args.teams}",
        f"--p1-network-path={glob.directory[lesserID.net_hash]}",
        f"--p2-network-path={glob.directory[greaterID.net_hash]}",
        f"--p1-search-time={args.search_iterations}",
        f"--p2-search-time={args.search_iterations}",
        f"--p1-bandit-name={lesserID.bandit_name}",
        f"--p2-bandit-name={greaterID.bandit_name}",
        f"--p1-policy-mode={lesserID.policy_mode}",
        f"--p2-policy-mode={greaterID.policy_mode}",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)
    last_line = result.stdout.strip().splitlines()[-1]
    data = list(map(int, last_line.split()))
    return lesserID, greaterID, data


def main():

    setup()

    while True:

        for _ in range(args.start, args.saves_per_update):

            print("step", _)

            args.start = 0

            sorted_ratings = sorted(glob.ratings.items(), key=lambda kv: kv[1])

            for key, value in sorted_ratings:
                if key in glob.ucb: 
                    print(
                        glob.directory[key.net_hash],
                        key.bandit_name,
                        key.policy_mode,
                        value,
                        glob.ucb[key],
                    )
            print("")

            with ThreadPoolExecutor(max_workers=args.threads) as pool:

                futures = []
                for _ in range(args.games_per_save):
                    futures.append(pool.submit(run_once))

                for fut in as_completed(futures):
                    lesserID, greaterID, data = fut.result()
                    glob.update(lesserID, greaterID, data)

            write_files()

        print(
            f"Removing lowest {args.n_delete} agents and replacing with random agents."
        )

        glob.remove_lowest(args.n_delete)
        glob.fill_from_path(args.net_path, args.n_delete)
        glob.reset_visits()


if __name__ == "__main__":
    main()

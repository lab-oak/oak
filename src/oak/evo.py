import sys
import os
import struct
import torch
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import subprocess
import math
import threading

import oak
import oak.torch

parser = argparse.ArgumentParser(description="Parameter testing for battle agents.")

parser.add_argument("--working-dir", default=None, type=str)
parser.add_argument("--net-path", default=".", type=str)
parser.add_argument("--budget", default=2**12, type=int)
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

class ID:
    def __init__(self, net_hash, bandit, policy_mode, iterations):
        self.net_hash = net_hash
        assert len(bandit) < 15, f"Error: bandit too long: {bandit}"
        self.bandit = bandit
        assert len(policy_mode) == 1, f"Error: policy mode must be char: {policy_mode}"
        self.policy_mode = policy_mode
        self.iterations = iterations

    def __eq__(self, other):
        if not isinstance(other, ID):
            return NotImplemented
        return (
            self.net_hash == other.net_hash
            and self.bandit == other.bandit
            and self.policy_mode == other.policy_mode
            and self.iterations == other.iterations
        )

    def __hash__(self):
        return hash((self.net_hash, self.bandit, self.policy_mode, self.iterations))

    def write(self, f):
        f.write(struct.pack("<Q", self.net_hash))
        bn = self.bandit.encode("utf8")
        bn = bn.ljust(15, b"\0")
        f.write(bn)
        f.write(self.policy_mode.encode("utf8"))
        f.write(struct.pack("<I", self.iterations))

    def print(self):
        print(self.net_hash, self.bandit, self.policy_mode, self.iterations)


def less_than(id1: ID, id2: ID):
    return (id1.net_hash, id1.bandit, id1.policy_mode, id1.iterations) < (
        id2.net_hash,
        id2.bandit,
        id2.policy_mode,
        id2.iterations,
    )


def read_id(f):
    data = f.read(8 + 15 + 1 + 4)
    if len(data) < 8 + 15 + 1 + 4:
        return None
    net_hash = struct.unpack("<Q", data[:8])[0]
    bandit = data[8:23].rstrip(b"\0").decode("ascii")
    mode = data[23:24].decode("ascii")
    iterations = struct.unpack("<I", data[24:28])[0]
    return ID(net_hash, bandit, mode, iterations)


class Elo:
    default_rating = 1200

    def __init__(self):
        self.table: Dict[ID, float] = {}

    def read(self, path):
        with open(path, "rb") as f:
            while True:
                id = read_id(f)
                if id is None:
                    break
                data = f.read(4)
                if len(data) < 4:
                    break
                rating = struct.unpack("<f", data)[0]
                self.table[id] = rating

    def write(self, path):
        with open(path, "wb") as f:
            for id, rating in self.table.items():
                id.write(f)
                f.write(struct.pack("<f", rating))


class UCB:
    def __init__(self):
        self.table: Dict[ID, tuple[float, int]] = {}

    def read(self, path, n=args.max_agents):
        temp = {}
        with open(path, "rb") as f:
            while True:
                id = read_id(f)
                if id is None:
                    break
                data = f.read(8)
                if len(data) < 8:
                    break
                score, visits = struct.unpack("<fI", data)
                temp[id] = [score, visits]
        sorted_ratings = sorted(temp.items(), key=lambda kv: kv[1], reverse=True)
        for key, value in sorted_ratings[:n]:
            self.table[key] = value

    def write(self, path):
        with open(path, "wb") as f:
            for id, data in self.table.items():
                id.write(f)
                f.write(struct.pack("<fI", data[0], data[1]))


class Results:
    def __init__(self):
        self.table: Dict[tuple[ID, ID], List[int, int, int]] = {}

    def read(self, path):
        with open(path, "rb") as f:
            while True:
                id1 = read_id(f)
                if id1 is None:
                    break
                id2 = read_id(f)
                if id2 is None:
                    break
                data = f.read(12)
                if len(data) < 12:
                    break
                w, l, d = struct.unpack("<III", data)
                self.table[(id1, id2)] = [w, l, d]

    def write(self, path):
        with open(path, "wb") as f:
            for (id1, id2), (w, l, d) in self.table.items():
                id1.write(f)
                id2.write(f)
                f.write(struct.pack("<III", w, l, d))


class NetworkDirectory:
    def __init__(self):
        self.table: Dict[int, str] = {}

    def fill_from_path(self, path):
        net_files = oak.util.find_data_files(path, ext=".battle.net")
        for file in net_files:
            network = oak.torch.BattleNetwork()
            with open(file, "rb") as f:
                network.read_parameters(f)
            self.table[network.hash()] = file
        self.table[0] = "mc"

    def read(self, path):
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
                self.table[net_hash] = b"".join(path_bytes).decode("utf-8")

    def write(self, path):
        with open(path, "wb") as f:
            for nh, path_str in self.table.items():
                f.write(struct.pack("<Q", nh))
                f.write(path_str.encode("utf-8"))
                f.write(b"\0")


class ProgramData:
    def __init__(self):
        self.elo = Elo()
        self.ucb = UCB()
        self.results = Results()

    def read(self, base):
        self.elo.read(os.path.join(base, "ratings"))
        self.ucb.read(os.path.join(base, "ucb"))
        self.results.read(os.path.join(base, "results"))

    def write(self, base):
        self.elo.write(os.path.join(base, "ratings"))
        self.ucb.write(os.path.join(base, "ucb"))
        self.results.write(os.path.join(base, "results"))


net_dir = NetworkDirectory()
program_data = ProgramData()

shutdown = threading.Event()
in_flight_lock = threading.Lock()
in_flight: Dict[int, tuple] = {}


def new_agent():
    net_hash, net_path = random.choice(list(net_dir.table.items()))

    base_bandits = ["exp3-", "ucb-"]
    if args.allow_policy:
        base_bandits.append("pexp3-")
        base_bandits.append("p2exp3-")
        base_bandits.append("pucb-")
    bandit = random.choice(base_bandits)

    param = None
    if (
        bandit.startswith("pexp3")
        or bandit.startswith("p2exp3")
        or bandit.startswith("exp3")
    ):
        param = random.uniform(args.exp3_param_min, args.exp3_param_max)
    elif bandit.startswith("pucb") or bandit.startswith("ucb"):
        param = random.uniform(args.ucb_param_min, args.ucb_param_max)
    else:
        assert False, "bad bandit name"
    bandit = bandit + str(param)[:5]

    selection_modes = ["n", "e", "x"]
    selection_mode = random.choice(selection_modes)

    return ID(net_hash, bandit, selection_mode, args.budget)


def fill_agents(n=args.max_agents):
    for _ in range(n):
        agent = new_agent()
        program_data.elo.table[agent] = Elo.default_rating
        program_data.ucb.table[agent] = [0, 0]


def select():
    N = sum(map(lambda kv: kv[1][1], program_data.ucb.table.items()))

    ids_sorted = sorted(
        program_data.ucb.table.items(),
        key=lambda kv: (kv[1][0]) + (args.c * math.sqrt(N) / (kv[1][1] + 1)),
        reverse=True,
    )
    if len(ids_sorted) < 2:
        raise RuntimeError("Need at least 2 IDs")

    first, second = ids_sorted[:2]

    first[1][1] = first[1][1] + 1
    second[1][1] = second[1][1] + 1

    id1 = first[0]
    id2 = second[0]
    if less_than(id2, id1):
        id1, id2 = id2, id1
    return id1, id2


def update(lesserID, greaterID, data):
    score = (data[0] + 0.5 * data[1]) / 2

    wdl = program_data.results.table.get((lesserID, greaterID), [0, 0, 0])
    wdl[0] += data[0]
    wdl[1] += data[1]
    wdl[2] += data[2]
    program_data.results.table[(lesserID, greaterID)] = wdl

    R_lesser = program_data.elo.table[lesserID]
    R_greater = program_data.elo.table[greaterID]

    E_lesser = 1 / (1 + 10 ** ((R_greater - R_lesser) / 400))
    E_greater = 1 - E_lesser

    program_data.elo.table[lesserID] += args.K * (score - E_lesser)
    program_data.elo.table[greaterID] += args.K * ((1 - score) - E_greater)

    v, n = program_data.ucb.table[lesserID]
    total_v = v * (n - 1)
    assert n > 0, "Bad UCB update"
    program_data.ucb.table[lesserID] = [(total_v + score) / n, n]
    v, n = program_data.ucb.table[greaterID]
    assert n > 0, "Bad UCB update"
    total_v = v * (n - 1)
    program_data.ucb.table[greaterID] = [(total_v + (1 - score)) / n, n]


def remove_lowest(n):
    sorted_ratings = sorted(program_data.ucb.table.items(), key=lambda kv: kv[1])
    for kv in sorted_ratings[:n]:
        del program_data.ucb.table[kv[0]]


def reset_visits():
    for key, value in program_data.ucb.table.items():
        value[1] = 0


def setup():
    if args.working_dir is None:
        import datetime

        now = datetime.datetime.now()
        working_dir = now.strftime("elo-%Y-%m-%d-%H:%M:%S")
        os.makedirs(working_dir, exist_ok=False)
        args.working_dir = working_dir

        net_dir.fill_from_path(args.net_path)
        fill_agents()
    else:
        print("Reading files")
        net_dir.read(os.path.join(args.working_dir, "directory"))
        program_data.read(args.working_dir)


def run_once() -> [ID, ID, [int, int, int]]:
    if shutdown.is_set():
        return None

    lesserID, greaterID = select()

    future_id = threading.get_ident()
    with in_flight_lock:
        in_flight[future_id] = (lesserID, greaterID)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "oak.vs",
        "--threads=1",
        "--max-games=1",
        "--skip-save",
        f"--teams={args.teams}",
        f"--p1-eval={net_dir.table[lesserID.net_hash]}",
        f"--p2-eval={net_dir.table[greaterID.net_hash]}",
        f"--p1-budget={lesserID.iterations}",
        f"--p2-budget={greaterID.iterations}",
        f"--p1-bandit={lesserID.bandit}",
        f"--p2-bandit={greaterID.bandit}",
        f"--p1-policy-mode={lesserID.policy_mode}",
        f"--p2-policy-mode={greaterID.policy_mode}",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    with in_flight_lock:
        in_flight.pop(future_id, None)

    last_line = result.stdout.strip().splitlines()[-1]
    data = list(map(int, last_line.split()))
    return lesserID, greaterID, data


def undo_in_flight():
    with in_flight_lock:
        for lesserID, greaterID in in_flight.values():
            if lesserID in program_data.ucb.table:
                program_data.ucb.table[lesserID][1] = max(
                    0, program_data.ucb.table[lesserID][1] - 1
                )
            if greaterID in program_data.ucb.table:
                program_data.ucb.table[greaterID][1] = max(
                    0, program_data.ucb.table[greaterID][1] - 1
                )
        in_flight.clear()


def run_batch(pool, n):
    futures = {}
    for _ in range(n):
        if shutdown.is_set():
            break
        f = pool.submit(run_once)
        futures[f] = True

    for fut in as_completed(futures):
        result = fut.result()
        if result is None:
            continue
        lesserID, greaterID, data = result
        update(lesserID, greaterID, data)

    return not shutdown.is_set()


def main():
    setup()

    try:
        while True:
            for _ in range(args.start, args.saves_per_update):
                print("step", _)
                args.start = 0

                sorted_ratings = sorted(
                    program_data.elo.table.items(), key=lambda kv: kv[1]
                )
                for key, value in sorted_ratings:
                    if key in program_data.ucb.table:
                        print(
                            net_dir.table[key.net_hash],
                            key.bandit,
                            key.policy_mode,
                            key.iterations,
                            value,
                            program_data.ucb.table[key],
                        )
                print("")

                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    completed = run_batch(pool, args.games_per_save)

                if shutdown.is_set():
                    break

                program_data.write(args.working_dir)

            if shutdown.is_set():
                break

            print(
                f"Removing lowest {args.n_delete} agents and replacing with random agents."
            )
            remove_lowest(args.n_delete)
            net_dir.fill_from_path(args.net_path)
            fill_agents(args.n_delete)
            reset_visits()

    except KeyboardInterrupt:
        shutdown.set()

    print("\nShutting down, waiting for in-flight games to finish...")
    undo_in_flight()
    program_data.write(args.working_dir)
    print("Saved.")


if __name__ == "__main__":
    main()

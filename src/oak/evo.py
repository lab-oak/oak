import sys
import os
import struct
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import subprocess
import math
import threading
import copy

import oak
import oak.torch

parser = argparse.ArgumentParser(description="Parameter testing for battle agents.")

parser.add_argument("--dir", default=None, type=str)
parser.add_argument("--network-dir", default=".", type=str)
parser.add_argument("--budget-ms", default=f"{2**5}", type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--max-agents", default=32, type=int)
parser.add_argument("--n-delete", default=8, type=int)
parser.add_argument("--games-per-save", default=2**10, type=int)
parser.add_argument("--saves-per-refresh", default=1, type=int)
parser.add_argument("--teams", default="", type=str)
parser.add_argument("--exp3-gamma-min", default=0.001, type=float)
parser.add_argument("--exp3-gamma-max", default=5.0, type=float)
parser.add_argument("--exp3-alpha-min", default=0.001, type=float)
parser.add_argument("--exp3-alpha-max", default=0.5, type=float)
parser.add_argument("--ucb-c-min", default=0.001, type=float)
parser.add_argument("--ucb-c-max", default=5.0, type=float)
parser.add_argument("--start", default=0, type=int)
# Agent selection UCB params
parser.add_argument("--c", default=1.414, type=float)
parser.add_argument("--K", default=8, type=float)

parser.add_argument("--no-exp3", action="store_true")
parser.add_argument("--no-pexp3", action="store_true")
parser.add_argument("--no-ucb", action="store_true")
parser.add_argument("--no-pucb", action="store_true")
parser.add_argument("--no-ucb1", action="store_true")

args = parser.parse_args()


class ID:
    def __init__(self, net_hash, bandit, policy_mode, ms, discrete=True):
        self.net_hash = net_hash
        assert len(bandit) < 15, f"Error: bandit too long: {bandit}"
        self.bandit = bandit
        assert len(policy_mode) == 1, f"Error: policy mode must be char: {policy_mode}"
        self.policy_mode = policy_mode
        self.ms = ms
        self.discrete = discrete

    def __eq__(self, other):
        if not isinstance(other, ID):
            return NotImplemented
        return (
            self.net_hash == other.net_hash
            and self.bandit == other.bandit
            and self.policy_mode == other.policy_mode
            and self.ms == other.ms
        )

    def __hash__(self):
        return hash((self.net_hash, self.bandit, self.policy_mode, self.ms))

    def write(self, f):
        f.write(struct.pack("<Q", self.net_hash))
        bn = self.bandit.encode("utf8")
        bn = bn.ljust(15, b"\0")
        f.write(bn)
        f.write(self.policy_mode.encode("utf8"))
        f.write(struct.pack("<I", self.ms))

    def print(self):
        print(self.net_hash, self.bandit, self.policy_mode, self.ms)


def permute_id(before: ID) -> ID:
    after = copy.deepcopy(before)

    choice = random.randint(0, 4)

    if choice == 0:
        bandit_type = after.bandit.split("-")[0]
        if bandit_type in ("exp3", "pexp3"):
            param = random.uniform(args.exp3_gamma_min, args.exp3_gamma_max)
        else:
            param = random.uniform(args.ucb_c_min, args.ucb_c_max)
        PRECISION = 5
        after.bandit = bandit_type + "-" + str(param)[:PRECISION]

    elif choice == 1:
        # New bandit type AND re-sample params
        base_bandits = ["exp3", "pexp3", "ucb", "pucb"]
        bandit_type = random.choice(base_bandits)
        if bandit_type in ("exp3", "pexp3"):
            param = random.uniform(args.exp3_gamma_min, args.exp3_gamma_max)
        else:
            param = random.uniform(args.ucb_c_min, args.ucb_c_max)
        PRECISION = 5
        after.bandit = bandit_type + "-" + str(param)[:PRECISION]

    elif choice == 2:
        net_hash, _ = random.choice(list(Options.network_table.items()))
        after.net_hash = net_hash

    elif choice == 3:
        modes = ["x", "n", "e"]
        modes.remove(after.policy_mode)
        after.policy_mode = random.choice(modes)
    elif choice == 4:
        # 50/50 flip discrete
        after.discrete = random.choice([True, False])
    else:
        assert False, "what"

    return after


def less_than(id1: ID, id2: ID):
    return (id1.net_hash, id1.bandit, id1.policy_mode, id1.ms) < (
        id2.net_hash,
        id2.bandit,
        id2.policy_mode,
        id2.ms,
    )


def read_id(f):
    data = f.read(8 + 15 + 1 + 4)
    if len(data) < 8 + 15 + 1 + 4:
        return None
    net_hash = struct.unpack("<Q", data[:8])[0]
    bandit = data[8:23].rstrip(b"\0").decode("ascii")
    mode = data[23:24].decode("ascii")
    ms = struct.unpack("<I", data[24:28])[0]
    return ID(net_hash, bandit, mode, ms)


class WDL:
    def __init__(self, w=0, d=0, l=0):
        self.win, self.draw, self.loss = (w, d, l)


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


class Options:
    network_table: Dict[int, str] = {}
    bandits = set()

    @staticmethod
    def fill_network_table(path):
        net_files = oak.util.find_data_files(path, ext=".battle.net")
        for file in net_files:
            # TODO remove
            step = int(file.split("/")[-1].split(".")[0])
            if (step % 500) or (step == 0):
                continue

            network = oak.torch.BattleNetwork()
            with open(file, "rb") as f:
                network.read_parameters(f)
            Options.network_table[network.hash()] = file
        # Options.network_table[0] = "mc"

    @staticmethod
    def read(path):
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
                Options.network_table[net_hash] = b"".join(path_bytes).decode("utf-8")

    @staticmethod
    def write(path):
        with open(path, "wb") as f:
            for nh, path_str in Options.network_table.items():
                f.write(struct.pack("<Q", nh))
                f.write(path_str.encode("utf-8"))
                f.write(b"\0")


class ProgramData:
    elo = Elo()
    ucb = UCB()
    results = Results()

    @staticmethod
    def read(base):
        ProgramData.elo.read(os.path.join(base, "ratings"))
        ProgramData.ucb.read(os.path.join(base, "ucb"))
        ProgramData.results.read(os.path.join(base, "results"))

    @staticmethod
    def write(base):
        ProgramData.elo.write(os.path.join(base, "ratings"))
        ProgramData.ucb.write(os.path.join(base, "ucb"))
        ProgramData.results.write(os.path.join(base, "results"))


shutdown = threading.Event()
in_flight_lock = threading.Lock()
in_flight: Dict[int, tuple] = {}


def new_agent():
    net_hash, network_path = random.choice(list(Options.network_table.items()))

    base_bandits = ["exp3", "ucb"]
    base_bandits.append("pexp3")
    base_bandits.append("pucb")
    bandit = random.choice(base_bandits)

    param = None
    if bandit.startswith("pexp3") or bandit.startswith("exp3"):
        param = random.uniform(args.exp3_gamma_min, args.exp3_gamma_max)
    elif (
        bandit.startswith("pucb")
        or bandit.startswith("ucb")
        or bandit.startswith("ucb1")
    ):
        param = random.uniform(args.ucb_c_min, args.ucb_c_max)
    else:
        assert False, "bad bandit name"
    PRECISION = 5
    bandit = bandit + "-" + str(param)[:PRECISION]

    selection_modes = ["n", "e", "x"]
    selection_mode = random.choice(selection_modes)

    return ID(net_hash, bandit, selection_mode, args.budget_ms)


def select():
    N = sum(map(lambda kv: kv[1][1], ProgramData.ucb.table.items()))

    ucb_sorted = sorted(
        ProgramData.ucb.table.items(),
        key=lambda kv: (kv[1][0]) + (args.c * math.sqrt(N) / (kv[1][1] + 1)),
        reverse=True,
    )
    if len(ucb_sorted) < 2:
        raise RuntimeError("Need at least 2 IDs")

    first, second = ucb_sorted[:2]

    # virtual
    for id_, ucb_entry in ucb_sorted[:2]:
        ucb_entry[1] += 1

    id1 = ucb_sorted[0][0]
    id2 = ucb_sorted[1][0]
    if less_than(id2, id1):
        id1, id2 = id2, id1
    return id1, id2


def update(lesserID, greaterID, wdl):
    score = (1.0 * wdl.win + 0.5 * wdl.draw + 0.0 * wdl.loss) / (
        wdl.win + wdl.draw + wdl.loss
    )

    # Results
    data = ProgramData.results.table.get((lesserID, greaterID), [0, 0, 0])
    data[0] += wdl.win
    data[1] += wdl.draw
    data[2] += wdl.loss
    ProgramData.results.table[(lesserID, greaterID)] = data

    # Elo
    R_lesser = ProgramData.elo.table[lesserID]
    R_greater = ProgramData.elo.table[greaterID]
    E_lesser = 1 / (1 + 10 ** ((R_greater - R_lesser) / 400))
    E_greater = 1 - E_lesser
    ProgramData.elo.table[lesserID] += args.K * (score - E_lesser)
    ProgramData.elo.table[greaterID] += args.K * ((1 - score) - E_greater)

    # UCB
    for i, id_ in enumerate([lesserID, greaterID]):
        v, virtual_n = ProgramData.ucb.table[id_]
        n = virtual_n - 1
        total_v = v * n
        update = score if i == 0 else (1 - score)
        v_ = (total_v + update) / (n + 1)
        ProgramData.ucb.table[id_] = [v_, virtual_n]


def setup():
    if args.dir is None:
        import datetime

        now = datetime.datetime.now()
        dir = now.strftime("elo-%Y-%m-%d-%H:%M:%S")
        os.makedirs(dir, exist_ok=False)
        args.dir = dir

        Options.fill_network_table(args.network_dir)
        for _ in range(args.max_agents):
            agent = new_agent()
            ProgramData.elo.table[agent] = Elo.default_rating
            ProgramData.ucb.table[agent] = [0, 0]
    else:
        print("Reading files")
        Options.read(os.path.join(args.dir, "directory"))
        ProgramData.read(args.dir)


def run_vs() -> [ID, ID, WDL]:
    if shutdown.is_set():
        return None

    lesserID, greaterID = select()

    future_id = threading.get_ident()
    with in_flight_lock:
        in_flight[future_id] = (lesserID, greaterID)

    cmd = [
        "vs",
        "--threads=1",
        "--max-games=1",
        "--mirror-match",
        f"--teams={args.teams}",
        f"--p1-eval={Options.network_table[lesserID.net_hash]}",
        f"--p2-eval={Options.network_table[greaterID.net_hash]}",
        f"--p1-budget={lesserID.ms}ms",
        f"--p2-budget={greaterID.ms}ms",
        f"--p1-bandit={lesserID.bandit}",
        f"--p2-bandit={greaterID.bandit}",
        f"--p1-policy-mode={lesserID.policy_mode}",
        f"--p2-policy-mode={greaterID.policy_mode}",
    ]

    if lesserID.discrete:
        cmd += ["--p1-use-discrete"]
    if greaterID.discrete:
        cmd += ["--p2-use-discrete"]

    result = subprocess.run(cmd, text=True, capture_output=True)

    with in_flight_lock:
        in_flight.pop(future_id, None)
    last_line = result.stdout.strip().splitlines()[-1]
    data = list(map(int, last_line.split()))
    wdl = WDL(*data)
    return lesserID, greaterID, wdl


def undo_in_flight():
    with in_flight_lock:
        for lesserID, greaterID in in_flight.values():
            if lesserID in ProgramData.ucb.table:
                ProgramData.ucb.table[lesserID][1] = max(
                    0, ProgramData.ucb.table[lesserID][1] - 1
                )
            if greaterID in ProgramData.ucb.table:
                ProgramData.ucb.table[greaterID][1] = max(
                    0, ProgramData.ucb.table[greaterID][1] - 1
                )
        in_flight.clear()


def run_batch(pool, n):
    futures = {}
    for _ in range(n):
        if shutdown.is_set():
            break
        f = pool.submit(run_vs)
        futures[f] = True

    for fut in as_completed(futures):
        result = fut.result()
        if result is None:
            continue
        lesserID, greaterID, wdl = result
        update(lesserID, greaterID, wdl)

    return not shutdown.is_set()


def refresh_agent_pool():
    top_agents = sorted(
        ProgramData.ucb.table.items(), key=lambda kv: kv[1][1], reverse=True
    )
    top_agents = [id_ for id_, _ in top_agents]

    for i in range(args.n_delete):
        parent = top_agents[i]
        child = copy.deepcopy(parent)
        while child in ProgramData.ucb.table:
            child = permute_id(parent)
        ProgramData.ucb.table[child] = [0, 0]
        ProgramData.elo.table[child] = Elo.default_rating

    for i in range(args.n_delete):
        id_ = top_agents[-(i + 1)]
        del ProgramData.ucb.table[id_]
        del ProgramData.elo.table[id_]

    print(f"Refresh: {len(ProgramData.ucb.table)} {len(ProgramData.elo.table)}")


def main():
    setup()

    try:
        while True:
            for _ in range(args.saves_per_refresh):
                sorted_ratings = sorted(
                    ProgramData.elo.table.items(), key=lambda kv: kv[1]
                )
                for key, value in sorted_ratings:
                    if key in ProgramData.ucb.table:
                        print(
                            Options.network_table[key.net_hash],
                            key.bandit,
                            key.policy_mode,
                            key.ms,
                            value,
                            ProgramData.ucb.table[key],
                        )
                print("")

                print("Run batch")
                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    completed = run_batch(pool, args.games_per_save)

                if shutdown.is_set():
                    break

                ProgramData.write(args.dir)

            if shutdown.is_set():
                break

            refresh_agent_pool()

    except KeyboardInterrupt:
        shutdown.set()

    print("\nShutting down, waiting for in-flight games to finish...")
    undo_in_flight()
    ProgramData.write(args.dir)
    print("Saved.")


if __name__ == "__main__":
    main()

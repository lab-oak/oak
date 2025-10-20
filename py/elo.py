import os
import torch
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import subprocess

import py_oak
import torch_oak


parser = argparse.ArgumentParser(description="Parameter testing for battle agents.")

parser.add_argument(
    "--working-dir",
    default=None,
    type=str,
)

parser.add_argument(
    "--vs-path",
    default="./release/vs",
    type=str,
)

parser.add_argument(
    "--search-iterations",
    default=2**12,
    type=int,
)

parser.add_argument(
    "--threads",
    default=1,
    type=int,
)

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

    def write(f):
        # write 8 byte hash, 15 char bytes, then 1 char byte to binary file stream
        pass


# global data


class Global:

    def __init__(
        self,
    ):

        # Elo ratings
        self.ratings: Dict[ID, int] = {}
        # UCB data
        self.ucb = Dict[ID, [float, int]]

        # W/D/L database
        self.results: Dict[tuple[ID, ID], tuple[int, int, int]] = {}

        self.directory: Dict[int, str] = {}

    def remove_id(id: ID):
        # search through add to remove all references. Results remove all entries that match either ID in the key

        pass

    def sample_ids() -> [ID, ID]:

        # compute ucb, sample argmax

        pass


glob = Global()


def read_files():
    with open(os.path.join([args.working_dir, "ratings"]), "rb") as f:
        # write ID then f32 elo rating
        pass

    with open(os.path.join([args.working_dir, "results"]), "rb") as f:
        # write ID1, ID2 (already ordered in key) then W, L, D u32s
        pass

    with open(os.path.join([args.working_dir, "directory"]), "rb") as f:
        # write 64 bit netowkr hash then null terminated path string
        pass


def write_files():
    # all of these should overwrite
    with open(os.path.join([args.working_dir, "ratings"]), "wb") as f:

        pass

    with open(os.path.join([args.working_dir, "results"]), "wb") as f:

        pass

    with open(os.path.join([args.working_dir, "directory"]), "wb") as f:

        pass


def setup():

    if args.working_dir is None:
        import datetime

        now = datetime.datetime.now()
        working_dir = now.strftime("battle-%Y-%m-%d-%H:%M:%S")
        os.makedirs(working_dir, exist_ok=False)
        args.working_dir = working_dir

    else:
        read_files()


def update():
    pass


def run_once(lesserID: ID, greaterID: ID):
    cmd = [
        args.vs_path,
        "--threads=1",
        "--game-pairs=1",
        f"--p1-network-path={glob.directory[lesserID.net_hash]}",
        f"--p2-network-path={glob.directory[greaterID.net_hash]}",
        f"--p1-search-time={args.search_iterations}",
        f"--p2-search-time={args.search_iterations}",
        f"--p1-bandit-name={lesserID.bandit_name}"
        f"--p2-bandit-name={greaterID.bandit_name}"
        f"--p1-policy-mode={lesserID.policy_mode}",
        f"--p2-policy-mode={greaterID.policy_mode}",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    return result


def main():

    setup()

    while True:

        with ThreadPoolExecutor(max_workers=args.threads) as pool:

            nullID = ID(0, "", "n")
            glob.directory[nullID.net_hash] = ""

            futures = [
                pool.submit(run_once(nullID, nullID))
                for i in range(args.games_per_update)
            ]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                print(f"[done {res['index']}] exit={res['exit_code']}")

    update()


if __name__ == "__main__":
    main()

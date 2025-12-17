import os
import argparse
import datetime
import subprocess
import signal
import threading
import sys
from typing import Dict

import py_oak


parser = argparse.ArgumentParser(
    description="Reinforcement learning using a generate process and battle.py (and optionally build.py)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


generate_parser = parser.add_argument_group("Data generation training arguments")
battle_parser = parser.add_argument_group("Battle network training arguments")
build_parser = parser.add_argument_group("Build network training arguments")


generate_parser.add_argument(
    "--generate-threads",
    type=int,
    default=(((os.cpu_count() or 2) - 1) or 1),
    help="Number of threads for self-play data generation",
)
generate_parser.add_argument(
    "--search-time",
    type=int,
    help="Number of iterations for training frames",
    required=True,
)
generate_parser.add_argument(
    "--fast-search-time",
    type=int,
    help="Number of iterations for non-training frames",
    required=True,
)
generate_parser.add_argument(
    "--fast-search-prob",
    type=float,
    help="Probability that fast-search-time is used instead of search-time",
    required=True,
)
generate_parser.add_argument(
    "--bandit-name",
    type=str,
    help="Bandit algorithm and parameters (exp3/pexp3/p2exp3/ucb/pucb)-(exp3 'gamma'/ucb 'c')",
    required=True,
)
generate_parser.add_argument(
    "--policy-mode", type=str, help="Mode for move selection (n/e/x/m)", required=True
)
generate_parser.add_argument(
    "--policy-nash-weight",
    type=float,
    help="Weight of nash policy when using mixed policy mode (m)",
)
generate_parser.add_argument(
    "--policy-min",
    type=float,
    default=0,
    help="Min prob of action before clamping to 0",
)
generate_parser.add_argument("--teams", type=str, default="", help="Path to teams file")

# get args from battle.py

import battle

battle.add_local_args(battle_parser, "", True)

import build

build.add_local_args(build_parser, "build", True)

parser.add_argument(
    "--generate-path",
    type=str,
    default="release/generate",
    help="Path to generate binary",
)
parser.add_argument(
    "--battle-path",
    type=str,
    default="py/battle.py",
    help="Path to battle.py script",
)
parser.add_argument(
    "--build-path",
    type=str,
    default="py/build.py",
    help="Path to build.py script",
)


def stream(prefix, pipe):
    for line in iter(pipe.readline, b""):
        print(f"[{prefix}] {line.decode()}", end="")
    pipe.close()


def main():

    args = parser.parse_args()

    use_build: bool = (args.team_modify_prob > 0) and (
        (args.pokemon_delete_prob > 0) or (args.move_delete_prob > 0)
    )

    assert (
        args.policy_mode != "m" or args.policy_nash_weight
    ), "Missing --policy-nash-weight while using (m)ixed -policy_mode"

    # core_battle_args = {}
    # network_path = None
    # dir = None
    # data_dir = .
    # steps = 0
    # in_place = False

    # [
    #     f"network-path={}",
    #     f"dir={}",
    #     f"data_dir={}",
    #     f"steps={}",
    #     "--in-place",
    # ]

    import torch
    import torch_oak

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    now = datetime.datetime.now()
    working_dir = now.strftime("rl-%Y-%m-%d-%H:%M:%S")
    os.makedirs(working_dir, exist_ok=False)
    py_oak.save_args(args, working_dir)

    network = torch_oak.BattleNetwork(
        args.pokemon_hidden_dim,
        args.active_hidden_dim,
        args.pokemon_out_dim,
        args.active_out_dim,
        args.hidden_dim,
        args.value_hidden_dim,
        args.policy_hidden_dim,
    )

    network_path = os.path.join(working_dir, "random.battle.net")
    with open(network_path, "wb") as f:
        network.write_parameters(f)

    data_dir = os.path.join(working_dir, "generate")
    nets_dir = os.path.join(working_dir, "nets")

    generate_cmd = [
        f"./{args.generate_path}",
        f"--threads={args.generate_threads}",
        f"--network-path={network_path}",
        f"--search-time={args.search_time}",
        f"--fast-search-time={args.fast_search_time}",
        f"--fast-search-prob={args.fast_search_prob}",
        f"--teams={args.teams}",
        f"--bandit-name={args.bandit_name}",
        f"--policy-mode={args.policy_mode}",
        f"--policy-min={args.policy_min}",
        f"--dir={data_dir}",
        f"--max-battle-length={args.max_battle_length}",
        "--buffer-size=1",
        "--keep-node=false",
    ]
    if args.no_apply_symmetries:
        generate_cmd.append("--no-apply-symmetries")
    if args.no_clamp_parameters:
        generate_cmd.append("--no-clamp-parameters")
    if args.policy_nash_weight is not None:
        generate_cmd.append(f"--policy-nash-weight={args.policy_nash_weight}")

    train_cmd = [
        f"{sys.executable}",
        "-u",
        f"{args.train_path}",
        f"--device={args.device}",
        f"--dir={nets_dir}",
        f"--threads={args.learn_threads}",
        f"--batch-size={args.batch_size}",
        f"--lr={args.lr}",
        f"--max-battle-length={args.max_battle_length}",
        f"--min-iterations={args.search_time}",
        f"--sleep={args.sleep}",
        f"--min-files={args.min_files}",
        f"--data-window={args.data_window}",
        f"--w-nash={args.w_nash}",
        f"--w-empirical={args.w_empirical}",
        f"--w-score={args.w_score}",
        f"--lr-decay={args.lr_decay}",
        f"--lr-decay-start={args.lr_decay_start}",
        f"--data-dir={data_dir}",
        f"--network-path={network_path}",
        f"--delete-window={args.delete_window}",
        "--in-place",
    ]

    exit()

    generate_proc = subprocess.Popen(
        generate_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
    )

    train_proc = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
    )

    threading.Thread(
        target=stream, args=("GENERATE", generate_proc.stdout), daemon=True
    ).start()
    threading.Thread(
        target=stream, args=("TRAIN", train_proc.stdout), daemon=True
    ).start()

    try:
        generate_proc.wait()
        train_proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl-C received, killing children...")

        for p in (generate_proc, train_proc):
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    print("Finished.")


if __name__ == "__main__":
    main()

import os
import argparse
import datetime
import subprocess
import signal
import threading
import sys

import py_oak

parser = argparse.ArgumentParser(
    description="Reinforcement learning using a generate process and battle.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--generate-path",
    type=str,
    default="release/generate",
    help="Path to generate binary",
)
parser.add_argument(
    "--train-path",
    type=str,
    default="py/battle.py",
    help="Path to battle.py script",
)

# Shared options
parser.add_argument(
    "--max-battle-length",
    type=int,
    default=2000,
    help="Max battle length in updates (not turns)",
)

# Worker options
parser.add_argument(
    "--generate-threads",
    type=int,
    default=(((os.cpu_count() or 2) - 1) or 1),
    help="Number of threads for self-play data generation",
)
parser.add_argument(
    "--search-time",
    type=int,
    help="Number of iterations for training frames",
    required=True,
)
parser.add_argument(
    "--fast-search-time",
    type=int,
    help="Number of iterations for non-training frames",
    required=True,
)
parser.add_argument(
    "--fast-search-prob",
    type=float,
    help="Probability that fast-search-time is used instead of search-time",
    required=True,
)
parser.add_argument("--teams", type=str, default="", help="Path to teams file")
parser.add_argument(
    "--bandit-name",
    type=str,
    help="Bandit algorithm and parameters (exp3/pexp3/p2exp3/ucb/pucb)-(exp3 'gamma'/ucb 'c')",
    required=True,
)
parser.add_argument(
    "--policy-mode", type=str, help="Mode for move selection (n/e/x/m)", required=True
)
parser.add_argument(
    "--policy-nash-weight",
    type=float,
    help="Weight of nash policy when using mixed policy mode (m)",
)
parser.add_argument(
    "--policy-min",
    type=float,
    default=0,
    help="Min prob of action before clamping to 0",
)

# General training
parser.add_argument("--batch-size", type=int, help="Batch size", required=True)
parser.add_argument(
    "--learn-threads",
    type=int,
    default=1,
    help="Number of threads for data loading/training",
)
parser.add_argument("--steps", type=int, default=0, help="Total training steps")
parser.add_argument(
    "--checkpoint", type=int, default=50, help="Checkpoint save interval (steps)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="Learning rate", required=True
)
parser.add_argument(
    "--no-clamp-parameters",
    action="store_true",
    help="Clamp parameters [-2, 2] to support Stockfish style quantization",
)
parser.add_argument(
    "--sleep",
    type=float,
    default=0,
    help="Number of seconds to sleep after parameter update",
)
parser.add_argument(
    "--min-files",
    type=int,
    default=1,
    help="Minimum number of .battle.data files before learning begins",
)

# Loss options
parser.add_argument(
    "--w-nash",
    type=float,
    default=0.0,
    help="Weight for Nash value in value target",
)
parser.add_argument(
    "--w-empirical",
    type=float,
    default=0.0,
    help="Weight for empirical value in value target",
)
parser.add_argument(
    "--w-score", type=float, default=1.0, help="Weight for score in value target"
)
parser.add_argument(
    "--w-nash-p",
    type=float,
    default=0.0,
    help="Weight for Nash in policy target (empirical = 1 - this)",
)
parser.add_argument(
    "--no-value-loss",
    action="store_true",
    dest="no_value_loss",
    help="Disable value loss computation",
)
parser.add_argument(
    "--no-policy-loss",
    action="store_true",
    dest="no_policy_loss",
    help="Disable policy loss computation",
)
parser.add_argument(
    "--w-policy-loss",
    type=float,
    default=1.0,
    help="Weight for policy loss relative to value loss",
)
parser.add_argument(
    "--lr-decay", type=float, default=1.0, help="Applied each step after decay begins"
)
parser.add_argument(
    "--lr-decay-start",
    type=int,
    default=0,
    help="The first step to begin applying lr decay",
)
parser.add_argument(
    "--lr-decay-interval",
    type=int,
    default=1,
    help="Interval at which to apply decay",
)
parser.add_argument(
    "--no-apply-symmetries",
    action="store_true",
    help="Whether to permute party Pokemon/Sides",
)
parser.add_argument(
    "--pokemon-hidden-dim",
    type=int,
    default=py_oak.pokemon_hidden_dim,
    help="Pokemon encoding net hidden dim",
)
parser.add_argument(
    "--active-hidden-dim",
    type=int,
    default=py_oak.active_hidden_dim,
    help="ActivePokemon encoding net hidden dim",
)
parser.add_argument(
    "--pokemon-out-dim",
    type=int,
    default=py_oak.pokemon_out_dim,
    help="Pokemon encoding net output dim",
)
parser.add_argument(
    "--active-out-dim",
    type=int,
    default=py_oak.active_out_dim,
    help="ActivePokemon encoding net output dim",
)
parser.add_argument(
    "--hidden-dim", type=int, default=py_oak.hidden_dim, help="Main subnet hidden dim"
)
parser.add_argument(
    "--value-hidden-dim",
    type=int,
    default=py_oak.value_hidden_dim,
    help="Value head hidden dim",
)
parser.add_argument(
    "--policy-hidden-dim",
    type=int,
    default=py_oak.policy_hidden_dim,
    help="Policy head hidden dim",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help='"cpu" or "cuda". Defaults to CUDA if available.',
)
parser.add_argument(
    "--print-window",
    type=int,
    default=5,
    help="Number of samples to print for debug output",
)
parser.add_argument(
    "--data-window",
    type=int,
    default=0,
    help="Only use the n-most recent files for freshness",
)
parser.add_argument(
    "--delete-window",
    type=int,
    default=0,
    help="Anything outside the most recent N files is deleted",
)

args = parser.parse_args()

assert (
    args.policy_mode != "m" or args.policy_nash_weight
), "Missing --policy-nash-weight while using (m)ixed -policy_mode"


def stream(prefix, pipe):
    for line in iter(pipe.readline, b""):
        print(f"[{prefix}] {line.decode()}", end="")
    pipe.close()


def main():

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

        # optional escalation
        for p in (generate_proc, train_proc):
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    print("Finished.")


if __name__ == "__main__":
    main()

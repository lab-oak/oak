import os
import torch
import argparse
import random
import time
import datetime
import subprocess

import py_oak
import torch_oak

parser = argparse.ArgumentParser(description="Train an Oak battle network.")

parser.add_argument(
    "--generate-path",
    type=str,
    default="./release/generate",
    help="Path to generate binary",
)
parser.add_argument(
    "--train-path",
    type=str,
    default="py/battle.py",
    help="Path to battle.py script",
)

# Shared options
parse.add_argument(
    "--max-battle-length",
    type=int,
    default=200,
    help="Max battle length"
)

# Worker options
parse.add_argument(
    "--search-time",
    type=str,
    default=2**12,
    help="Full search time"
)
parse.add_argument(
    "--fast-search-time",
    type=str,
    default=2**10,
    help="Full search time"
)
parse.add_argument(
    "--fast-search-prob",
    type=float,
    default=7/8,
    help="Probability that only a fast search is used"
)
parse.add_argument(
    "--teams",
    type=str,
    default="",
    help="Path to teams file"
)
parse.add_argument(
    "--bandit-name",
    type=str,
    default="p2exp3-.1",
    help="Bandit algorithm and parameters"
)
parse.add_argument(
    "--policy-mode",
    type=str,
    default="m",
    help="Mode for move selection"
)
parse.add_argument(
    "--policy-nash-weight",
    type=float,
    default=.9,
    help="Weight of nash policy when using mixed policy mode (default)"
)
parse.add_argument(
    "--policy-min",
    type=float,
    default=.001,
    help="Min prob of action before clamping to 0"
)

# General training
parser.add_argument("--batch-size", default=2**10, type=int, help="Batch size")
parser.add_argument(
    "--self-play-threads", type=int, default=(os.cpu_count() - 1), help="Number of threads for self-play data generation"
)
parser.add_argument(
    "--learn-threads", type=int, default=1, help="Number of threads for data loading/training"
)
parser.add_argument("--steps", type=int, default=2**30, help="Total training steps")
parser.add_argument(
    "--checkpoint", type=int, default=50, help="Checkpoint interval (steps)"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--max-battle-length",
    type=int,
    default=10000,
    help="Ignore games past this length (in updates not turns.)",
)
parser.add_argument(
    "--clamp-parameters",
    type=bool,
    default=True,
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
    "--apply-symmetries",
    type=bool,
    default=True,
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


def main():

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    now = datetime.datetime.now()
    working_dir = now.strftime("battle-%Y-%m-%d-%H:%M:%S")
    os.makedirs(working_dir, exist_ok=False)
    py_oak.save_args(args, working_dir)

    generate_cmd = [
        "",
        "--threads=1",
        "--max-games=1",
        "--skip-save",
        f"--teams={args.teams}",
        f"--p1-network-path={glob.directory[lesserID.net_hash]}",
        f"--p2-network-path={glob.directory[greaterID.net_hash]}",
        f"--p1-search-time={lesserID.iterations}",
        f"--p2-search-time={greaterID.iterations}",
        f"--p1-bandit-name={lesserID.bandit_name}",
        f"--p2-bandit-name={greaterID.bandit_name}",
        f"--p1-policy-mode={lesserID.policy_mode}",
        f"--p2-policy-mode={greaterID.policy_mode}",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)



if __name__ == "__main__":
    main()

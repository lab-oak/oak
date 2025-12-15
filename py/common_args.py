import argparse
import time
from typing import List

import py_oak

def add_common_args(parser: argparse.ArgumentParser, prefix: str = ""):

    if prefix:
        prefix = prefix + "-"
    prefix = "--" + prefix

    # Program
    parser.add_argument(
        prefix + "device",
        type=str,
        help='"cpu" or "cuda". Defaults to CUDA if available.',
    )
    parser.add_argument(
        prefix + "threads",
        type=int,
        default=1,
        help="Number of threads for data loading/training",
    )
    parser.add_argument(
        prefix + "network-path", type=str, help="Path for initial network weights"
    )
    parser.add_argument(
        prefix + "dir",
        type=str,
        help="Output directory. Timestamp if not provided",
    )
    parser.add_argument(
        prefix + "data-dir",
        default=".",
        help="Read directory for data files. All subdirectories are scanned",
    )

    # Basic Training
    parser.add_argument(
        prefix + "batch-size", required=True, type=int, help="Batch size"
    )
    parser.add_argument(prefix + "lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        prefix + "lr-decay",
        type=float,
        default=1.0,
        help="Applied each step after decay begins",
    )
    parser.add_argument(
        prefix + "lr-decay-start",
        type=int,
        default=0,
        help="The first step to begin applying lr decay",
    )
    parser.add_argument(
        prefix + "lr-decay-interval",
        type=int,
        default=1,
        help="Interval at which to apply decay",
    )
    parser.add_argument(
        prefix + "steps",
        type=int,
        default=0,
        help="Total training steps. A value of 0 is treated as infinity",
    )

    # Train/Generate interop
    parser.add_argument(
        prefix + "in-place",
        action="store_true",
        help="The parameters saved in --network-path will be updated after every step",
    )
    parser.add_argument(
        prefix + "data-window",
        type=int,
        default=0,
        help="Only use the n-most recent files for freshness",
    )
    parser.add_argument(
        prefix + "min-files",
        type=int,
        default=1,
        help="Minimum number of data files before learning begins",
    )
    parser.add_argument(
        prefix + "sleep",
        type=float,
        default=0,
        help="Number of seconds to sleep after parameter update",
    )

    # QoL
    parser.add_argument(
        prefix + "checkpoint", type=int, default=50, help="Checkpoint interval (steps)"
    )
    parser.add_argument(
        prefix + "delete-window",
        type=int,
        default=0,
        help="Anything outside the most recent N files is deleted",
    )
    parser.add_argument(prefix + "seed", type=int, help="Random seed for determinism")


def namespace_to_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[str]:
    result = []

    for action in parser._actions:
        if not action.option_strings:
            continue  # positional args

        dest = action.dest
        value = getattr(args, dest)

        # pick the long option if it exists
        opt = next(
            (s for s in action.option_strings if s.startswith("--")),
            action.option_strings[0],
        )

        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(opt)

        elif isinstance(action, argparse._StoreAction):
            if value is not None:
                result.append(f"{opt}={value}")

        # add more action types here if needed

    return result

def get_files(args : argparse.ArgumentParser, ext : str) -> [List[str], bool] :
    data_files = py_oak.find_data_files(args.data_dir, ext)

    if len(data_files) < args.min_files:
        print("Minimum files not reached. Sleeping")
        time.sleep(5)
        return data_files, False

    if args.delete_window > 0:
        to_delete = data_files[args.delete_window :]
        for file in to_delete:
            os.remove(file)

    if args.data_window > 0:
        data_files = data_files[: args.data_window]

    return data_files, True

def save_and_decay(args : argparse.ArgumentParser, step : int):
    if step >= args.lr_decay_start:
        if (step % args.lr_decay_interval) == 0:
            for group in opt.param_groups:
                group["lr"] *= args.lr_decay

    if ((step + 1) % args.checkpoint) == 0:
        ckpt_path = ""
        if args.in_place:
            ckpt_path = args.network_path
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
        ckpt_path = os.path.join(args.dir, f"{step + 1}.battle.net")
        with open(ckpt_path, "wb") as f:
            network.write_parameters(f)
        print(f"Checkpoint saved at step {step + 1}: {ckpt_path}")

    time.sleep(args.sleep)


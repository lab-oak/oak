import argparse


def add_common_args(parser: argparse.ArgumentParser, prefix: str = ""):

    prefix = "--" + prefix
    if prefix:
        prefix = prefix + "-"

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

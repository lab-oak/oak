import os
import torch
import argparse
import random

import py_oak
import torch_oak

parser = argparse.ArgumentParser(description="Train an Oak battle network.")
parser.add_argument(
    "--in-place",
    action="store_true",
    dest="in_place",
    help="Used for RL.",
)
parser.add_argument("--net-path", default="", help="Read directory for network weights")
parser.add_argument(
    "--data-dir",
    default=".",
    help="Read directory for battle files. All subdirectories are scanned.",
)
parser.add_argument("--batch-size", default=2**10, type=int, help="Batch size")

# General training
parser.add_argument(
    "--threads", type=int, default=1, help="Number of threads for data loading"
)
parser.add_argument("--steps", type=int, default=2**16, help="Total training steps")
parser.add_argument(
    "--checkpoint", type=int, default=500, help="Checkpoint interval (steps)"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--write-prob",
    type=float,
    default=1 / 100,
    help="Write probability for encode_buffers. A lower value means less correlated samples.",
)
parser.add_argument(
    "--max-battle-length",
    type=int,
    default=10000,
    help="Ignore games past this length (in updates not turns.)",
)
parser.add_argument(
    "--min-iterations",
    type=int,
    default=1,
    help="Ignore frames with fewer than these iterations.",
)
parser.add_argument(
    "--clamp-parameters",
    type=bool,
    default=True,
    help="Clamp parameters [-2, 2] to support Stockfish style quantization",
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
    default=1.0,
    help="Weight for empirical value in value target",
)
parser.add_argument(
    "--w-score", type=float, default=0.0, help="Weight for score in value target"
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
    help=".",
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

# Network hyperparameters
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

# Hardware and reproducibility
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed for determinism"
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help='"cpu" or "cuda". Defaults to CUDA if available.',
)

# Debug / printing
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

args = parser.parse_args()


# TODO accept other optims as arg
class Optimizer:
    def __init__(self, network: torch.nn.Module, lr):
        self.opt = torch.optim.Adam(network.parameters(), lr=lr)

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


def masked_kl_div(logit, target):
    log_probs = torch.log_softmax(logit, dim=-1)
    log_probs = log_probs.masked_fill(torch.isneginf(log_probs), 0)
    log_target = torch.log(target)
    log_target = log_target.masked_fill(torch.isneginf(log_target), 0)
    kl = target * (log_target - log_probs)
    # kl = kl.masked_fill(torch.isneginf(logit), 0)
    return kl.sum(dim=1).mean(dim=0)


def masked_cross_entropy(logit, target):
    log_probs = torch.log_softmax(logit, dim=-1)
    log_probs = log_probs.masked_fill(torch.isneginf(log_probs), 0)
    ce = -target * log_probs
    return ce.sum(dim=1).mean(dim=0)


def loss(
    input: torch_oak.EncodedBattleFrames,
    output: torch_oak.OutputBuffers,
    args,
    print_flag=False,
):
    size = min(input.size, output.size)

    # Value target
    w_nash = args.w_nash
    w_empirical = args.w_empirical
    w_score = args.w_score

    assert (w_nash + w_empirical + w_score) == 1, "value target weights don't sum to 1"
    value_target = (
        w_nash * input.nash_value[:size]
        + w_empirical * input.empirical_value[:size]
        + w_score * input.score[:size]
    )

    # Policy target
    w_nash_p = args.w_nash_p
    w_empirical_p = 1 - w_nash_p
    assert (w_nash_p + w_empirical_p) == 1

    p1_policy_target = (
        w_empirical_p * input.p1_empirical[:size] + w_nash_p * input.p1_nash[:size]
    )
    p2_policy_target = (
        w_empirical_p * input.p2_empirical[:size] + w_nash_p * input.p2_nash[:size]
    )

    loss = torch.zeros((1,))

    if not args.no_value_loss:
        loss += torch.nn.functional.mse_loss(output.value[:size], value_target)
    value_loss = loss.detach().clone()

    if not args.no_policy_loss:
        p1_policy_loss = masked_cross_entropy(output.p1_policy[:size], p1_policy_target)
        p2_policy_loss = masked_cross_entropy(output.p2_policy[:size], p2_policy_target)
        loss += args.w_policy_loss * (p1_policy_loss + p2_policy_loss)

    if print_flag:
        window = args.print_window
        print(
            torch.cat(
                [
                    output.value[:window],
                    value_target[:window],
                    input.empirical_value[:window],
                    input.nash_value[:window],
                    input.score[:window],
                ],
                dim=1,
            )
        )
        if not args.no_policy_loss:
            print("P1 policy inference/target")
            x = torch.nn.functional.softmax(output.p1_policy[:window], 1).view(
                window, 1, 9
            )
            y = p1_policy_target[:window].view(window, 1, 9)
            print(torch.cat([x, y], dim=1))
            print(f"loss: p1:{p1_policy_loss}, p2:{p2_policy_loss}")
        if not args.no_value_loss:
            print(f"loss: v:{value_loss.mean()}")

    return loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Device
    if args.seed is not None:
        set_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    working_dir = ""
    if args.in_place:
        assert args.net_path
    else:
        import datetime

        now = datetime.datetime.now()
        working_dir = now.strftime("battle-%Y-%m-%d-%H:%M:%S")
        os.makedirs(working_dir, exist_ok=False)

    network = torch_oak.BattleNetwork(
        args.pokemon_hidden_dim,
        args.active_hidden_dim,
        args.pokemon_out_dim,
        args.active_out_dim,
        args.hidden_dim,
        args.value_hidden_dim,
        args.policy_hidden_dim,
    )
    if args.net_path:
        with open(args.net_path, "rb") as f:
            network.read_parameters(f)

    data_files = py_oak.find_data_files(args.data_dir, ext=".battle.data")

    sample_indexer = py_oak.SampleIndexer()

    for file in data_files:
        sample_indexer.get(file)

    print("Saving base network in working dir.")
    with open(os.path.join(working_dir, "random.battle.net"), "wb") as f:
        network.write_parameters(f)

    if len(data_files) == 0:
        print(
            f"No .battle.data files found in {args.data_dir}. Run ./release/generate with appropriate options to make them."
        )
        exit()
    else:
        print(f"{len(data_files)} data_files found")

    encoded_frames = py_oak.EncodedBattleFrames(args.batch_size)
    encoded_frames_torch = torch_oak.EncodedBattleFrames(encoded_frames)
    output_buffer = torch_oak.OutputBuffers(
        args.batch_size, args.pokemon_out_dim, args.active_out_dim
    )
    optimizer = Optimizer(network, args.lr)

    for step in range(args.steps):
        data_files = py_oak.find_data_files(args.data_dir, ext=".battle.data")
        if args.data_window > 0:
            data_files = data_files[: args.data_window]
        encoded_frames.clear()
        output_buffer.clear()

        py_oak.encode_buffers_2(sample_indexer, encoded_frames, args.threads, args.max_battle_length, args.min_iterations)

        # apply symmetries for more varied data
        encoded_frames_torch.permute_pokemon()
        encoded_frames_torch.permute_sides()

        network.inference(encoded_frames_torch, output_buffer, not args.no_policy_loss)

        optimizer.zero_grad()
        loss_value = loss(
            encoded_frames_torch, output_buffer, args, (step % args.checkpoint) == 0
        )
        loss_value.backward()
        optimizer.step()

        if args.clamp_parameters:
            network.clamp_parameters()

        if step >= args.lr_decay_start:
            for group in optimizer.opt.param_groups:
                group["lr"] *= args.lr_decay

        if ((step + 1) % args.checkpoint) == 0:
            ckpt_path = ""
            if args.in_place:
                ckpt_path = args.net_path
            else:
                ckpt_path = os.path.join(working_dir, f"{step + 1}.battle.net")
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
            print(f"Checkpoint saved at step {step + 1}: {ckpt_path}")


if __name__ == "__main__":
    main()

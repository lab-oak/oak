import os
import argparse
import random
import time
import datetime
import itertools

import py_oak

parser = argparse.ArgumentParser(
    description="Train an Oak battle network.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

import common_args

common_args.add_common_args(parser)

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
    help="Ignore samples with fewer than these iterations.",
)
parser.add_argument(
    "--no-clamp-parameters",
    action="store_true",
    help="Clamp parameters [-2, 2] to support Stockfish style quantization",
)
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
    "--no-apply-symmetries",
    action="store_true",
    help="Whether to skip permuting Bench/Sides",
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
parser.add_argument(
    "--print-window",
    type=int,
    default=5,
    help="Number of samples to print for debug output",
)


def main():

    args = parser.parse_args()

    assert (
        not args.in_place or args.network_path
    ), "--network-path must be provided when --in-place is used."

    import torch
    import torch_oak

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

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

        assert (
            w_nash + w_empirical + w_score
        ) == 1, "value target weights don't sum to 1"
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
            p1_policy_loss = masked_cross_entropy(
                output.p1_policy[:size], p1_policy_target
            )
            p2_policy_loss = masked_cross_entropy(
                output.p2_policy[:size], p2_policy_target
            )
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

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.dir is None:
        now = datetime.datetime.now()
        args.dir = now.strftime("battle-%Y-%m-%d-%H:%M:%S")

    os.makedirs(args.dir, exist_ok=False)
    py_oak.save_args(args, args.dir)

    network = torch_oak.BattleNetwork(
        args.pokemon_hidden_dim,
        args.active_hidden_dim,
        args.pokemon_out_dim,
        args.active_out_dim,
        args.hidden_dim,
        args.value_hidden_dim,
        args.policy_hidden_dim,
    ).to(device)

    if args.network_path:
        with open(args.network_path, "rb") as f:
            network.read_parameters(f)

    with open(os.path.join(args.dir, "initial.battle.net"), "wb") as f:
        network.write_parameters(f)
        print("Saved initial network in output directory.")

    encoded_frames = py_oak.EncodedBattleFrames(args.batch_size)
    encoded_frames_torch = torch_oak.EncodedBattleFrames(encoded_frames).to(device)

    output_buffer = torch_oak.OutputBuffers(
        args.batch_size, args.pokemon_out_dim, args.active_out_dim
    ).to(device)

    optimizer = Optimizer(network, args.lr)

    step_iterator = range(args.steps) if args.steps > 0 else itertools.count()

    skipped_steps = 0

    for s in step_iterator:

        step = s - skipped_steps

        data_files, enough = common_args.get_files(args)
        if not enough:
            skipped_steps += 1
            continue

        # TODO wasteful
        sample_indexer = py_oak.SampleIndexer()

        for file in data_files:
            sample_indexer.get(file)

        encoded_frames.clear()
        output_buffer.clear()

        samples_read = py_oak.encode_buffers_2(
            sample_indexer,
            encoded_frames,
            args.threads,
            args.max_battle_length,
            args.min_iterations,
        )

        if samples_read < args.batch_size:\
            skipped_steps += 1
            print("Error during sampling, continuing...")
            continue

        if not args.no_apply_symmetries:
            encoded_frames_torch.permute_pokemon()
            encoded_frames_torch.permute_sides()

        network.inference(encoded_frames_torch, output_buffer, not args.no_policy_loss)

        optimizer.zero_grad()
        loss_value = loss(
            encoded_frames_torch, output_buffer, args, (step % args.checkpoint) == 0
        )
        loss_value.backward()
        optimizer.step()

        if not args.no_clamp_parameters:
            network.clamp_parameters()

        common_args.save_and_decay(args, step)

if __name__ == "__main__":
    main()

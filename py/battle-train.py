import os
import torch
import argparse
import random

import py_oak
import torch_oak


def print_tensors(obj):
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            print(
                f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
            )
            print(value)


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
    kl = target * (torch.log(target) - log_probs)
    kl = kl.masked_fill(torch.isneginf(logit), 0)
    return kl.sum(dim=1).mean(dim=0)


def loss(
    input: torch_oak.BattleFrame,
    output: torch_oak.OutputBuffers,
    args,
    print_flag=False,
):
    size = min(input.size, output.size)

    # Value target
    value_target = (
        args.w_nash * input.nash_value[:size]
        + args.w_empirical * input.empirical_value[:size]
        + args.w_score * input.score[:size]
    )

    # Policy target
    w_nash_p = args.w_nash_p
    w_empirical_p = 1 - w_nash_p

    p1_policy_target = (
        w_empirical_p * input.p1_empirical[:size] + w_nash_p * input.p1_nash[:size]
    )
    p2_policy_target = (
        w_empirical_p * input.p2_empirical[:size] + w_nash_p * input.p2_nash[:size]
    )

    # Loss
    value_loss = torch.nn.functional.mse_loss(output.value[:size], value_target)

    if args.no_policy_loss:
        p1_loss = torch.tensor(0.0, device=value_loss.device)
        p2_loss = torch.tensor(0.0, device=value_loss.device)
    else:
        p1_loss = masked_kl_div(output.p1_policy[:size], p1_policy_target)
        p2_loss = masked_kl_div(output.p2_policy[:size], p2_policy_target)

    total_loss = value_loss + p1_loss + p2_loss

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
            x = torch.nn.functional.softmax(output.p1_policy[:window], 1).view(
                window, 1, 9
            )
            y = p1_policy_target[:window].view(window, 1, 9)
            print(torch.cat([x, y], dim=1))
        print(f"loss: v:{value_loss.mean()}, p1:{p1_loss}, p2:{p2_loss}")

    return total_loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train an Oak battle network.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="Used for RL.",
    )
    parser.add_argument(
        "--net-dir", default="", help="Write directory for network weights"
    )
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

    # Loss weighting
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
        default=1.0,
        help="Weight for Nash in policy target (empirical = 1 - this)",
    )

    # Policy loss toggle
    parser.add_argument(
        "--no-policy-loss",
        action="store_true",
        dest="no_policy_loss",
        help="Disable policy loss computation",
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
        working_dir = now.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(working_dir, exist_ok=False)
        print(f"Created working dir: {working_dir}")

    data_files = py_oak.find_data_files(args.data_dir, ext=".battle")
    print(f"{len(data_files)} files found")

    encoded_frames = py_oak.EncodedBattleFrame(args.batch_size)
    encoded_frames_torch = torch_oak.BattleFrame(encoded_frames)
    output_buffer = torch_oak.OutputBuffers(args.batch_size)
    network = torch_oak.BattleNetwork().to(args.device)
    optimizer = Optimizer(network, args.lr)

    for step in range(args.steps):
        data_files = py_oak.find_data_files(args.data_dir, ext=".battle")
        if args.data_window > 0:
            data_files = data_files[: args.data_window]
        encoded_frames.clear()
        output_buffer.clear()

        py_oak.encode_buffers(
            data_files,
            args.threads,
            args.batch_size,
            encoded_frames,
            start_index=0,
            write_prob=args.write_prob,
        )

        # apply symmetries for more varied data
        encoded_frames_torch.permute_pokemon()
        encoded_frames_torch.permute_sides()

        network.inference(encoded_frames_torch, output_buffer)

        optimizer.zero_grad()
        loss_value = loss(
            encoded_frames_torch, output_buffer, args, (step % args.checkpoint) == 0
        )
        loss_value.backward()
        optimizer.step()

        network.clamp_parameters()

        if ((step + 1) % args.checkpoint) == 0:
            ckpt_path = ""
            if args.in_place:
                ckpt_path = args.net_path
            else:
                ckpt_path = os.path.join(working_dir, f"{step + 1}.net")
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
            print(f"Checkpoint saved at step {step + 1}: {ckpt_path}")


if __name__ == "__main__":
    main()

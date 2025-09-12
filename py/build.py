import os
import torch
import argparse
import random

import py_oak
import torch_oak

parser = argparse.ArgumentParser(description="Train an Oak build network with PPO.")
# Keep updating the same file so that `generate` can use an up to date network
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
parser.add_argument("--steps", type=int, default=2**16, help="Total training steps")
parser.add_argument(
    "--checkpoint", type=int, default=100, help="Checkpoint interval (steps)"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--keep-prob",
    type=float,
    default=1.0,
    help=".",
)

# Learning
parser.add_argument(
    "--batch-size",
    type=int,
    default=2**10,
)
parser.add_argument(
    "--value-weight",
    type=float,
    default=1.0,
    help="Value vs Score weighting.",
)
parser.add_argument(
    "--entropy-weight",
    type=float,
    default=0.01,
)

parser.add_argument(
    "--data-window",
    type=int,
    default=0,
    help="Only use the n-most recent files for freshness",
)

# Device
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed for determinism"
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help='"cpu" or "cuda". Defaults to CUDA if available.',
)

args = parser.parse_args()


# Turn [b, T, 1] actions into [b, T, N] state
def get_state(actions):
    b, T, _ = actions.shape
    state = torch.zeros(
        (b, T, py_oak.species_move_list_size + 1), dtype=torch.float32
    )  # [b, T, N+1]
    state = state.scatter(2, actions + 1, 1.0)
    state = torch.cumsum(state, dim=1).clamp_max(1.0)
    state = state[:, :-1, 1:]  # [b, T, N]
    state = torch.concat([torch.zeros(state[:, :1].shape), state], dim=1)
    return state


class PPO:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2


def process_targets(
    network: torch_oak.BuildNetwork,
    traj: torch_oak.BuildTrajectory,
    ppo: PPO,
    value_weight=1,
):
    b, T, _ = traj.mask.shape
    # only do up to max traj length
    T = torch.max(traj.end).item()

    valid_actions = traj.actions != -1  # [b, T, 1]
    valid_choices = traj.policy > 0  # [b, T, 1]
    valid = torch.logical_and(valid_actions, valid_choices).squeeze(-1)

    state = get_state(traj.actions)
    valid_state = state[valid]
    logits, v = network.forward(valid_state)
    value = torch.sigmoid(v)

    # [v, 339]
    valid_mask = traj.mask[valid]
    mask_weights = torch.where(
        valid_mask >= 0,
        torch.exp(torch.gather(logits, 1, torch.clamp(valid_mask, min=0))),
        torch.zeros_like(valid_mask),
    )
    mask_probs = torch.nn.functional.normalize(mask_weights, dim=1, p=1)

    # [v, 1]
    valid_logp = torch.log(mask_probs[:, 0]).unsqueeze(-1)
    valid_old_logp = torch.log(traj.policy[valid])
    valid_ratio = torch.exp(valid_logp - valid_old_logp)

    # [b, T, 1]
    ratio = torch.ones_like(traj.policy)
    ratio[valid] = valid_ratio
    score_weight = 1 - args.value_weight
    r = value_weight * traj.value + score_weight * traj.score
    rewards = torch.zeros_like(traj.policy).scatter(
        1, traj.end.unsqueeze(-1) - 1, r.unsqueeze(-1)
    )
    value_full = torch.zeros_like(traj.policy)
    value_full[valid] = value
    next_value_full = torch.cat(
        [value_full[:, 1:], torch.zeros_like(value_full[:, -1:])], dim=1
    )
    deltas = rewards + ppo.gamma * next_value_full - value_full

    advantages = []
    advantage = torch.zeros((b, 1))
    for t in reversed(range(T)):
        # Only update advantage if step is valid
        advantage = (
            deltas[:, t, :] * valid[:, t].unsqueeze(-1)
            + ppo.gamma * ppo.lam * advantage
        )
        advantages.insert(0, advantage)
    gae = torch.stack(advantages, dim=1)  # [b, T, 1]
    returns = gae + value_full

    surr1 = ratio * gae
    surr2 = torch.clamp(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * gae
    surr = torch.min(surr1, surr2)
    logp = torch.ones_like(traj.policy)
    logp[valid] = valid_logp
    data = (
        surr,
        returns,
        value_full,
        logp,
    )
    valid_data = tuple(x[valid] for x in data)
    return valid_data


def compute_loss_from_targets(
    surr, returns, values, logp_actions, total, remaining, entropy_weight=0.01
) -> torch.Tensor:
    b = surr.shape[0]
    n = min(b, remaining)
    policy_loss = -(surr[:n]).mean()
    value_loss = (((returns - values)) ** 2)[:n].mean()
    entropy = -(logp_actions * logp_actions.exp()).sum(dim=-1, keepdim=True)[:n].mean()
    loss = (n / total) * (policy_loss + 0.5 * value_loss - entropy_weight * entropy)
    return loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

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

    network = torch_oak.BuildNetwork()
    if args.net_path:
        with open(args.net_path, "rb") as f:
            network.read_parameters(f)

    data_files = py_oak.find_data_files(args.data_dir, ext=".build")
    print("Saving base network in working dir.")
    with open(os.path.join(working_dir, "build-network"), "wb") as f:
        network.write_parameters(f)

    if len(data_files) == 0:
        print(
            f"No .build files found in {args.data_dir}. Run ./build/generate with appropriate options to make them."
        )
        exit()
    else:
        print(f"{len(data_files)} data_files found")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    steps = args.steps
    batch_size = args.batch_size
    keep_prob = args.keep_prob

    ppo = PPO()
    from random import sample

    for step in range(steps):
        data_files = py_oak.find_data_files(args.data_dir, ext=".build")
        if args.data_window > 0:
            data_files = data_files[: args.data_window]

        print("step:", step)
        optimizer.zero_grad()
        b = 0

        # break batches up by file to limit memory use
        while b < batch_size:
            file = sample(data_files, 1)[0]
            trajectories = py_oak.read_build_trajectories(file)

            T = trajectories.end.max()
            # here is where we trunacte the episode length for 1v1, etc
            traj = torch_oak.BuildTrajectory(trajectories, n=T)

            surr, returns, values, logp = process_targets(
                network, traj, ppo, args.value_weight
            )

            mask = torch.rand(surr.shape) < keep_prob
            surr, returns, values, logp = (
                surr[mask],
                returns[mask],
                values[mask],
                logp[mask],
            )
            if surr.numel() == 0:
                print("empty targets after sampling, probably a small buffer")
                continue

            loss = compute_loss_from_targets(
                surr,
                returns,
                values,
                logp,
                batch_size,
                batch_size - b,
                args.entropy_weight,
            )
            loss.backward()

            b += surr.shape[0]

        optimizer.step()

        if ((step + 1) % args.checkpoint) == 0:
            ckpt_path = ""
            if args.in_place:
                ckpt_path = args.net_path
            else:
                ckpt_path = os.path.join(working_dir, f"{step}.net")
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")


if __name__ == "__main__":
    main()

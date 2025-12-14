import os
import torch
import argparse
import random
import time


parser = argparse.ArgumentParser(
    description="Train an Oak build network.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

import common_args

common_args.add_common_args(parser)

parser.add_argument(
    "--keep-prob",
    type=float,
    default=1.0,
    help=".",
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


def main():

    args = parser.parse_args()

    assert (
        not args.in_place or args.network_path
    ), "--network-path must be provided when --in-place is used."

    import torch
    import torch_oak

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

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
        traj: torch_oak.BuildTrajectories,
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
        entropy = (
            -(logp_actions * logp_actions.exp()).sum(dim=-1, keepdim=True)[:n].mean()
        )
        loss = (n / total) * (policy_loss + 0.5 * value_loss - entropy_weight * entropy)
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
        args.dir = now.strftime("build-%Y-%m-%d-%H:%M:%S")

    os.makedirs(args.dir, exist_ok=False)
    py_oak.save_args(args, args.dir)

    network = torch_oak.BuildNetwork()

    if args.network_path:
        with open(args.network_path, "rb") as f:
            network.read_parameters(f)

    with open(os.path.join(args.dir, "initial.build.net"), "wb") as f:
        network.write_parameters(f)
        print("Saved initial network in output directory.")

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    ppo = PPO()

    step_iterator = range(args.steps) if args.steps > 0 else itertools.count()

    for step in step_iterator:
        data_files = py_oak.find_data_files(args.data_dir, ext=".build.data")
        if args.data_window > 0:
            data_files = data_files[: args.data_window]

        optimizer.zero_grad()
        b = 0

        # break batches up by file to limit memory use
        while b < args.batch_size:
            file = random.sample(data_files, 1)[0]
            trajectories = py_oak.read_build_trajectories(file)

            T = trajectories.end.max()
            # here is where we trunacte the episode length for 1v1, etc
            traj = torch_oak.BuildTrajectories(trajectories, n=T)

            surr, returns, values, logp = process_targets(
                network, traj, ppo, args.value_weight
            )

            mask = torch.rand(surr.shape) < args.keep_prob
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
                args.batch_size,
                args.batch_size - b,
                args.entropy_weight,
            )
            loss.backward()

            b += surr.shape[0]

        optimizer.step()

        if ((step + 1) % args.checkpoint) == 0:
            ckpt_path = ""
            if args.in_place:
                ckpt_path = args.network_path
            else:
                ckpt_path = os.path.join(working_dir, f"{step}.build.net")
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

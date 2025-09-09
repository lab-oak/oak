import net
import torch
import pyoak
import os


def find_data_files(root_dir, ext=".battle"):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


def get_state(actions):
    b, T, _ = actions.shape
    state = torch.zeros(
        (b, T, pyoak.species_move_list_size + 1), dtype=torch.float32
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
    network: net.BuildNetwork, traj: net.BuildTrajectoryTorch, ppo: PPO
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
    rewards = torch.zeros_like(traj.policy).scatter(
        1, traj.end.unsqueeze(-1) - 1, traj.value.unsqueeze(-1)
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


def compute_loss_from_targets(surr, returns, values, logp_actions):
    policy_loss = -(surr).mean()

    value_loss = (((returns - values)) ** 2).mean()

    entropy = -(logp_actions * logp_actions.exp()).sum(dim=-1, keepdim=True).mean()

    loss = policy_loss + 0.5 * value_loss
    return loss


def main():
    import sys

    if len(sys.argv) < 4:
        print("Input: in-net-path, out-net-path, steps")

    files = find_data_files(".", ext=".build")
    assert len(files) > 0, "No build files found in cwd"

    network = net.BuildNetwork()

    with open(sys.argv[1], "rb") as f:
        network.read_parameters(f)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    steps = int(sys.argv[3])
    batch_size = 2**14
    keep_prob = 1

    ppo = PPO()

    from random import sample

    for _ in range(steps):
        print("step:", _)

        targets = [torch.zeros((0,)) for _ in range(4)]

        b = 0
        while b < batch_size:

            file = sample(files, 1)[0]
            trajectories = pyoak.read_build_trajectories(file)
            T = trajectories.end.max()
            traj = net.BuildTrajectoryTorch(trajectories, n=T)

            new_targets = process_targets(network, traj, ppo)
            r = torch.rand(new_targets[0].shape)
            new_targets = tuple(t[r < keep_prob] for t in new_targets)
            targets = tuple(
                torch.cat([old, new], dim=0) for old, new in zip(targets, new_targets)
            )
            b = targets[0].shape[0]
            print(b)

        optimizer.zero_grad()
        l = compute_loss_from_targets(*targets)
        l.backward()
        optimizer.step()

    with open(sys.argv[2], "wb") as f:
        network.write_parameters(f)


if __name__ == "__main__":
    main()

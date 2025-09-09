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
        self.gamma = .99
        self.lam = .95
        self.clip_eps = .2

def process_targets(network : net.BuildNetwork, traj : net.BuildTrajectoryTorch, ppo : PPO):
    b, T, _ = traj.mask.shape

    valid_actions = traj.actions != -1  # [b, T, 1]
    valid_choices = traj.policy > 0  # [b, T, 1]
    valid = torch.logical_and(valid_actions, valid_choices).squeeze(-1)

    state = get_state(traj.actions)
    valid_state = state[valid]
    logits, value = network.forward(valid_state)

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
    ratio = torch.ones_like(traj.actions, dtype=torch.float)
    ratio[valid] = valid_ratio
    rewards = torch.zeros_like(traj.actions, dtype=torch.float).scatter(
        1, traj.end.unsqueeze(-1) - 1, traj.value.unsqueeze(-1)
    )

    value_full = torch.zeros_like(traj.policy)
    value_full[valid] = value
    for x in value_full[:10]:
        print(x)


    return

    # --- GAE ---
    gamma, lam = 0.99, 0.95
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
    deltas = rewards + gamma * next_values - values

    advantages = []
    advantage = torch.zeros((b, 1), device=values.device)
    for t in reversed(range(T)):
        # Only update advantage if step is valid
        advantage = deltas[:, t, :] * valid_mask[:, t, :] + gamma * lam * advantage
        advantages.insert(0, advantage)

    gae = torch.stack(advantages, dim=1)  # [b, T, 1]
    returns = gae + values

    ratios = torch.exp(logp_actions - old_logp_actions)

    clip_eps = 0.2
    surr1 = ratios * gae
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * gae
    return (
        torch.min(surr1, surr2)[valid_mask == 1],
        returns[valid_mask == 1],
        values[valid_mask == 1],
        logp_actions[valid_mask == 1],
    )


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
    batch_size = 2**12
    keep_prob = 1

    ppo = PPO()

    from random import sample

    for _ in range(steps):
        print(_)

        targets = [torch.zeros((0,)) for _ in range(4)]

        b = 0
        while b < batch_size:

            file = sample(files, 1)[0]
            traj = net.BuildTrajectoryTorch(pyoak.read_build_trajectories(file))

            new_targets = process_targets(network, traj, ppo)
            exit()
            r = torch.rand(new_targets[0].shape)
            new_targets = [t[r < keep_prob] for t in new_targets]
            targets = [
                torch.cat([old, new], dim=0) for old, new in zip(targets, new_targets)
            ]
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

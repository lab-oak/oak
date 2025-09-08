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
    safe_actions = torch.from_numpy(actions + 1).long()
    state = state.scatter(2, safe_actions, 1.0)
    state = torch.cumsum(state, dim=1).clamp_max(1.0)
    state = state[:, :-1, 1:]  # [b, T, N]
    state = torch.concat([torch.zeros(state[:, :1].shape), state], dim=1)
    return state


def process_targets(values, logits, trajectories):
    b, T, _ = trajectories.mask.shape

    traj = net.BuildTrajectoryTorch(trajectories)

    valid_actions = traj.actions != -1  # [b, T, 1]
    valid_choices = traj.policy > 0  # [b, T, 1]
    valid_mask = torch.logical_and(valid_actions, valid_choices).float()

    logits_flat = logits.view(-1, pyoak.species_move_list_size)  # [(b*T), N]
    actions_flat = torch.clamp(traj.actions.view(-1), min=0)  # [(b*T)]
    mask0_flat = torch.clamp(traj.mask[:, :, 0].view(-1), min=0)  # [(b*T)]

    # gather the values to swap
    logits_a = logits_flat[torch.arange(b * T), actions_flat]
    logits_m = logits_flat[torch.arange(b * T), mask0_flat]

    # swap them without in-place assignment
    swapped_logits = logits_flat.clone()
    swapped_logits = swapped_logits.scatter(
        1, actions_flat.unsqueeze(1), logits_m.unsqueeze(1)
    )
    swapped_logits = swapped_logits.scatter(
        1, mask0_flat.unsqueeze(1), logits_a.unsqueeze(1)
    )

    # reshape back
    logits = swapped_logits.view(b, T, -1)

    safe_actions = torch.clamp(traj.mask, min=0)
    mask_logits = torch.gather(logits, 2, safe_actions)
    mask_logits = torch.exp(mask_logits)

    # zero out invalid mask entries (out-of-place)
    mask_logits = torch.where(
        traj.mask == -1, torch.zeros_like(mask_logits), mask_logits
    )

    # normalize into probabilities
    mask_probs = torch.nn.functional.normalize(mask_logits, dim=2)
    p_actions = mask_probs[:, :, 0]

    logp_actions = torch.log(p_actions).unsqueeze(-1) * valid_mask
    old_logp_actions = torch.log(traj.policy) * valid_mask

    # replace NaNs safely
    logp_actions = torch.nan_to_num(logp_actions, nan=1.0)
    old_logp_actions = torch.nan_to_num(old_logp_actions, nan=1.0)

    value_weight = 1
    rewards_flat = value_weight * traj.value + (1 - value_weight) * traj.score
    rewards = torch.zeros_like(logp_actions)
    end_expanded = traj.end.unsqueeze(-1) - 1
    rewards = rewards.scatter(
        1, end_expanded, rewards_flat.unsqueeze(-1)
    )  # out-of-place

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


def loss(values, logits, build_trajectories):
    b, T, _ = build_trajectories.mask.shape

    traj = net.BuildTrajectoryTorch(build_trajectories)

    valid_actions = traj.actions != -1  # [b, T, 1]
    valid_choices = traj.policy > 0  # [b, T, 1]
    valid_mask = torch.logical_and(valid_actions, valid_choices).float()

    logits_flat = logits.view(-1, pyoak.species_move_list_size)  # [(b*T), N]
    actions_flat = torch.clamp(traj.actions.view(-1), min=0)  # [(b*T)]
    mask0_flat = torch.clamp(traj.mask[:, :, 0].view(-1), min=0)  # [(b*T)]

    # gather the values to swap
    logits_a = logits_flat[torch.arange(b * T), actions_flat]
    logits_m = logits_flat[torch.arange(b * T), mask0_flat]

    # swap them without in-place assignment
    swapped_logits = logits_flat.clone()
    swapped_logits = swapped_logits.scatter(
        1, actions_flat.unsqueeze(1), logits_m.unsqueeze(1)
    )
    swapped_logits = swapped_logits.scatter(
        1, mask0_flat.unsqueeze(1), logits_a.unsqueeze(1)
    )

    # reshape back
    logits = swapped_logits.view(b, T, -1)

    safe_actions = torch.clamp(traj.mask, min=0)
    mask_logits = torch.gather(logits, 2, safe_actions)
    mask_logits = torch.exp(mask_logits)

    # zero out invalid mask entries (out-of-place)
    mask_logits = torch.where(
        traj.mask == -1, torch.zeros_like(mask_logits), mask_logits
    )

    # normalize into probabilities
    mask_probs = torch.nn.functional.normalize(mask_logits, dim=2)
    p_actions = mask_probs[:, :, 0]

    logp_actions = torch.log(p_actions).unsqueeze(-1) * valid_mask
    old_logp_actions = torch.log(traj.policy) * valid_mask

    # replace NaNs safely
    logp_actions = torch.nan_to_num(logp_actions, nan=1.0)
    old_logp_actions = torch.nan_to_num(old_logp_actions, nan=1.0)

    value_weight = 1
    r = value_weight * traj.value + (1 - value_weight) * traj.score

    rewards = torch.zeros_like(logp_actions)
    end_expanded = traj.end.unsqueeze(-1) - 1
    rewards = rewards.scatter(1, end_expanded, r.unsqueeze(-1))  # out-of-place

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
    policy_loss = -(
        torch.min(surr1, surr2) * valid_mask
    ).sum() / valid_mask.sum().clamp_min(1.0)

    value_loss = (
        ((returns - values) * valid_mask) ** 2
    ).sum() / valid_mask.sum().clamp_min(1.0)

    # entropy = (
    #     -(logp_actions * logp_actions.exp()).sum(dim=-1, keepdim=True) * valid_mask
    # ).sum() / valid_mask.sum().clamp_min(1.0)

    loss = policy_loss + 0.5 * value_loss
    return loss


def main():

    files = find_data_files(".", ext=".build")
    assert len(files) > 0, "No build files found in cwd"

    network = net.BuildNetwork()

    with open("./1/build-network", "rb") as f:
        network.read_parameters(f)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    steps = 100
    batch_size = 2**12
    keep_prob = 1

    from random import sample

    for _ in range(steps):
        print(_)

        targets = [torch.zeros((0,)) for _ in range(4)]

        b = 0
        while b < batch_size:

            file = sample(files, 1)[0]
            build_trajectories = pyoak.read_build_trajectories(file)

            state = get_state(build_trajectories.actions)

            logits, v = network.forward(state)
            values = torch.sigmoid(v)

            new_targets = process_targets(values, logits, build_trajectories)
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

    with open("./build-network", "wb") as f:
        network.write_parameters(f)


if __name__ == "__main__":
    main()

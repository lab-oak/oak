import sys
import os

import pyoak
import numpy as np


def find_data_files(root_dir, ext=".battle"):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


def read_build_trajectories():

    files = find_data_files(".", ext=".build")
    assert len(files) > 0, "No build files found in cwd"

    from random import sample

    file = sample(files, 1)[0]
    build_trajectories = pyoak.read_build_trajectories(file)

    assert build_trajectories.size > 0, f"No data found in {file}."

    for i in range(min(10, build_trajectories.size)):
        index = sample(list(range(build_trajectories.size)), 1)[0]
        print(f"Sample {index}:")
        species_move = [
            pyoak.species_move_list[_]
            for _ in build_trajectories.actions[index].reshape(-1)
        ]
        species_move_string = [
            (pyoak.species_names[s], pyoak.move_names[m]) for s, m in species_move
        ]
        selection_probs = [
            float(_) for _ in build_trajectories.policy[index].reshape(-1)
        ]

        data = [(sm, p) for sm, p in zip(species_move_string, selection_probs) if p > 0]
        print(data)
        print(build_trajectories.value[index])
        print(build_trajectories.score[index])

    import net
    import torch

    b, T, _ = build_trajectories.mask.shape
    N = pyoak.species_move_list_size

    traj = net.BuildTrajectoryTorch(build_trajectories)

    valid_actions = traj.actions != -1  # [b, T, 1]
    valid_choices = traj.policy > 0
    valid_mask = torch.logical_and(valid_actions, valid_choices).float()

    value_network = net.EmbeddingNet(
        pyoak.species_move_list_size, 512, 1, True, False
    )

    policy_network = net.EmbeddingNet(
        pyoak.species_move_list_size, 512, pyoak.species_move_list_size, True, False
    )

    def get_state(actions):
        state = torch.zeros((b, T, N + 1), dtype=torch.float32)  # [b, T, N+1]
        safe_actions = torch.from_numpy(actions + 1).long()
        state = state.scatter(2, safe_actions, 1.0)
        state = torch.cumsum(state, dim=1).clamp_max(1.0)
        state = state[:, :-1, 1:]  # [b, T, N]
        state = torch.concat([torch.zeros(state[:, :1].shape), state], dim=1)
        return state

    state = get_state(build_trajectories.actions)


    values = torch.sigmoid(value_network.forward(state))
    logits = policy_network.forward(state)

    logits_flat = logits.view(-1, pyoak.species_move_list_size)  # [(b*T), N]
    actions_flat = traj.actions.view(-1)        # [(b*T)]
    mask0_flat = traj.mask[:, :, 0].view(-1)   # [(b*T)]

    # gather the values to swap
    logits_a = logits_flat[torch.arange(b*T), actions_flat]
    logits_m = logits_flat[torch.arange(b*T), mask0_flat]

    # swap them
    logits_flat[torch.arange(b*T), actions_flat] = logits_m
    logits_flat[torch.arange(b*T), mask0_flat] = logits_a

    # reshape back
    logits = logits_flat.view(b, T, -1)

    safe_actions = torch.clamp(traj.mask, min=0)
    mask_logits = torch.gather(logits, 2, safe_actions)
    mask_logits = torch.exp(mask_logits)
    # mask_logits[invalid_mask] = 0
    mask_logits[traj.mask == -1] = 0
    # mask_probs = mask_logits / torch.sum(mask_logits, dim=2)
    mask_probs = torch.nn.functional.normalize(mask_logits, dim=2)
    p_actions = mask_probs[:, :, 0]
    logp_actions = torch.log(p_actions).unsqueeze(-1) * valid_mask
    old_logp_actions = torch.log(traj.policy) * valid_mask
    logp_actions[torch.isnan(logp_actions)] = 1
    old_logp_actions[torch.isnan(old_logp_actions)] = 1

    value_weight = 1
    r = value_weight * traj.value + (1 - value_weight) * traj.score

    rewards = torch.zeros_like(logp_actions)
    end_expanded = traj.end.unsqueeze(-1) - 1
    rewards.scatter_(1, end_expanded, r.unsqueeze(-1))

    # --- GAE ---
    gamma, lam = 0.99, 0.95
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
    deltas = rewards + gamma * next_values - values

    advantages = []
    advantage = torch.zeros((b, 1))
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







def create_set():

    from net import EmbeddingNet

    network = EmbeddingNet(
        pyoak.species_move_list_size, 512, pyoak.species_move_list_size, True, False
    )

    if len(sys.argv) < 3:
        print("no build network path provided; using randomly initialized net")
    else:
        with open(sys.argv[2]) as params:
            network.read_parameters(params)

    import torch

    team = torch.zeros([pyoak.species_move_list_size])

    # create mask for choosing the first species
    mask = torch.zeros([pyoak.species_move_list_size])
    for index, pair in enumerate(pyoak.species_move_list):
        s, m = pair
        if m == 0:
            mask[index] = 1

    def sample_masked_logits(logits, mask):
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return sampled

    index = sample_masked_logits(network.forward(team), mask)
    species, _ = pyoak.species_move_list[index]
    print(pyoak.species_names[species])
    team[index] = 1

    # reset mask and fill with legal moves
    mask.zero_()
    n_moves = 0
    for index, pair in enumerate(pyoak.species_move_list):
        s, m = pair
        if s == species and m != 0:
            n_moves += 1
            mask[index] = 1

    for _ in range(min(4, n_moves)):
        index = sample_masked_logits(network.forward(team), mask)
        _, move = pyoak.species_move_list[index]
        print(pyoak.move_names[move])
        team[index] = 1
        mask[index] = 0


if __name__ == "__main__":

    key = sys.argv[1]

    if key == "read-build-trajectories":
        # print the first 10 trajectories in cwd
        read_build_trajectories()
    elif key == "create-set":
        # recreates the build networking rollout code to create a single pokemon set
        create_set()
    else:
        print("Invalid keyword. See TUTORIAL.md")

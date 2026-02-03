import sys

import pyoak
import numpy as np


def battle_frame_stats():
    total_frames = 0
    total_battles = 0
    path = "."
    if len(sys.argv) >= 3:
        path = sys.argv[2]

    files = pyoak.find_data_files(path, ext=".battle.data")
    for file in files:
        data = pyoak.read_battle_data(file)
        for buf, n in data:
            total_battles += 1
            total_frames += n
    print(f"Total battle frames: {total_frames}")
    print(f"Average battle length: {total_frames / total_battles}")


def read_battle_trajectories():

    path = "."
    if len(sys.argv) >= 3:
        path = sys.argv[2]

    # using only the head gives most recent files
    files = pyoak.find_data_files(path, ext=".battle.data")

    assert len(files) > 0, f"No battle files found in {path}"

    from random import sample, randint, shuffle

    file = sample(files, 1)[0]
    # file = files[0]

    data = pyoak.read_battle_data(file)

    shuffle(data)

    for buf, n in data:
        frames = pyoak.get_encoded_frames(buf, n)

        # i = randint(0, frames.size - 1)
        i = -1
        print(i, frames.m[i].item(), frames.n[i].item())

        print("score", frames.score[i])
        print("value", frames.empirical_value[i])
        print("e1", frames.p1_empirical[i])
        print("e2", frames.p2_empirical[i])
        print("n1", frames.p1_nash[i])
        print("n2", frames.p2_nash[i])
        print(
            "c1",
            [pyoak.policy_dim_labels[_.item()] for _ in frames.p1_choice_indices[i]],
        )
        print(
            "c2",
            [pyoak.policy_dim_labels[_.item()] for _ in frames.p2_choice_indices[i]],
        )
        raw_frames = pyoak.get_frames(buf, n)

        print("iterations", frames.iterations[i])

        # print(raw_frames.battle[i])
        # print(raw_frames.battle[i, 23])
        # print(raw_frames.durations[i])
        pyoak.print_battle_data(raw_frames, i)


def read_build_trajectories():

    # using only the head gives most recent files
    files = pyoak.find_data_files(".", ext=".build.data")
    assert len(files) > 0, "No build files found in cwd"

    from random import sample

    build_trajectories, read = pyoak.read_build_trajectories(files, 1024, 1)
    print(build_trajectories.size, read)
    assert build_trajectories.size == read, f"Bad read from {file}."
    for i in range(min(10, build_trajectories.size)):
        index = sample(list(range(build_trajectories.size)), 1)[0]
        print(f"Sample {index}:")
        species_move = [
            pyoak.species_move_list[_]
            for _ in build_trajectories.actions[index].reshape(-1)
        ]
        names = []
        for sm in species_move:
            s, m = sm
            names.append(pyoak.species_names[s] + " " + pyoak.move_names[m])
        selection_probs = [
            int(1000 * float(_)) / 10
            for _ in build_trajectories.policy[index].reshape(-1)
        ]
        actions = build_trajectories.actions[index].reshape(-1)

        data = [
            f"{n}:{p}" for n, p, a in zip(names, selection_probs, actions) if a >= 0
        ]

        l = 31
        print(data)
        print("value", build_trajectories.value[index].item())
        print("score", build_trajectories.score[index].item())
        # continue
        print("actions", build_trajectories.actions[index, :l])
        print("legal actions mask", build_trajectories.mask[index, :l, :20])
        print("log probs", np.log(build_trajectories.policy[index, :l]))
        print(
            "start/end",
            np.concatenate(
                [
                    build_trajectories.start[index, :l],
                    build_trajectories.end[index, :l],
                ],
                axis=0,
            ),
        )


def show_species_probs():
    import torch_oak
    import torch
    import math

    network = torch_oak.BuildNetwork()

    path = sys.argv[2]
    with open(path, "rb") as file:
        network.read_parameters(file)

    weights = dict()
    logits_d = dict()

    logits, _ = network.forward(torch.zeros((1, pyoak.species_move_list_size)))

    for index, pair in enumerate(pyoak.species_move_list):
        s, m = pair
        if m != 0:
            continue

        name = pyoak.species_names[s]
        weights[name] = math.exp(logits[0, index])
        logits_d[name] = logits[0, index]

    s = 0
    for x in weights:
        s += weights[x]
    probs = [(species, weights[species] / s, logits_d[species]) for species in weights]
    probs = sorted(probs, key=lambda x: x[1])
    for x in probs:
        print(x[0], int(1000 * x[1]) / 1000, x[2].item())


def create_set():

    from torch_oak import BuildNetwork

    network = BuildNetwork()

    if len(sys.argv) < 3:
        print("no build network path provided; using randomly initialized net")
    else:
        with open(sys.argv[2], "rb") as params:
            network.read_parameters(params)

    n = 1
    if len(sys.argv) >= 4:
        n = int(sys.argv[3])

    import torch

    for _ in range(n):
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
            probs_p = torch.softmax(masked_logits * 1, dim=-1)
            sampled = torch.multinomial(probs_p, 1)
            return (
                sampled,
                int(probs[sampled].item() * 10000) / 100,
                int(probs_p[sampled].item() * 10000) / 100,
            )

        logits, _ = network.forward(team)
        index, p, q = sample_masked_logits(logits, mask)
        species, _ = pyoak.species_move_list[index]
        print(f"{pyoak.species_names[species]} : {p}% ~ {q}%")
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
            index, p, q = sample_masked_logits(network.forward(team)[0], mask)
            _, move = pyoak.species_move_list[index]
            print(f"    {pyoak.move_names[move]} : {p}% ~ {q}%")
            team[index] = 1
            mask[index] = 0


def test_consistency():

    if len(sys.argv) < 4:
        print("Provide path to battle network and data file to test")
        exit()

    import torch
    import pyoak
    import torch_oak

    network_path = sys.argv[2]
    data_path = sys.argv[3]

    network = torch_oak.BattleNetwork()

    with open(network_path, "rb") as f:
        network.read_parameters(f)

    buffer_list = pyoak.read_battle_data(data_path)

    max_games = 2

    for buffer, n_frames in buffer_list[:max_games]:

        encoded_frames = pyoak.get_encoded_frames(buffer, n_frames)
        encoded_frames_torch = torch_oak.EncodedBattleFrames(encoded_frames)

        output = torch_oak.OutputBuffers(encoded_frames.size)

        network.inference(encoded_frames_torch, output)

        print(output.value)
        # policies = torch.cat([output.p1_policy.unsqueeze(1), output.p2_policy.unsqueeze(1)], dim=1)

    pyoak.test_consistency(max_games, network_path, data_path)


if __name__ == "__main__":

    key = sys.argv[1]

    if key == "read-build-trajectories":
        # print the first 10 trajectories in cwd
        read_build_trajectories()
    elif key == "read-battle-trajectories":
        read_battle_trajectories()
    elif key == "create-set":
        # recreates the build networking rollout code to create a single pokemon set
        create_set()
    elif key == "show-species-probs":
        # basic check that PPO works. We expect to see less ratata and more snorlax no matter what
        show_species_probs()
    elif key == "battle-frame-stats":
        battle_frame_stats()
    elif key == "test-consistency":
        test_consistency()
    else:
        print("Invalid keyword. See TUTORIAL.md")

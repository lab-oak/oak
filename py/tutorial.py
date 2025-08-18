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
            pyoak.species_move_list[_] for _ in build_trajectories.actions[index].reshape(-1)
        ]
        species_move_string = [
            (pyoak.species_names[s], pyoak.move_names[m]) for s, m in species_move
        ]
        selection_probs = [float(_) for _ in build_trajectories.policy[index].reshape(-1)]

        data = [(sm, p) for sm, p in zip(species_move_string, selection_probs) if p > 0]
        print(data)
        print(build_trajectories.eval[index])
        print(build_trajectories.score[index])


def create_set():

    from net import EmbeddingNet

    network = EmbeddingNet(
        pyoak.species_move_list_size, 512, pyoak.species_move_list_size, True, False
    )

    if (len(sys.argv) < 3):
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
        if (s == species and m != 0):
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

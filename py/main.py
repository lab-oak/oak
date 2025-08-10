import os

import read


def find_battle_files(root_dir):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".battle"):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


def main():
    total = 0
    files = find_battle_files("/home/user/train/2025-07-14-18:30:58/")
    for file in files:
        battle_data = read.read_battle_data(file)
        # battle_frames = [libread.get_frames(*pee) for pee in battle_data]        # battle_frames = [libread.get_frames(*pee) for pee in battle_data]
        encoded_frames = read.get_encoded_frames(*battle_data[0])

        turn = 73

        p1 = [
            (
                read.policy_dim_labels[i],
                int(e.item() * 1000) / 10,
                int(n.item() * 1000) / 10,
            )
            for e, n, i in zip(
                encoded_frames.p1_empirical[turn],
                encoded_frames.p1_nash[turn],
                encoded_frames.p1_choice_indices[turn],
            )
        ]
        p2 = [
            (
                read.policy_dim_labels[i],
                int(e.item() * 1000) / 10,
                int(n.item() * 1000) / 10,
            )
            for e, n, i in zip(
                encoded_frames.p2_empirical[turn],
                encoded_frames.p2_nash[turn],
                encoded_frames.p2_choice_indices[turn],
            )
        ]
        print("P1")
        for p in p1:
            if not p[0]:
                break
            print(p)
        print("P2")
        for p in p2:
            if not p[0]:
                break
            print(p)

        print(
            "value:",
            encoded_frames.empirical_value[turn].item(),
            encoded_frames.nash_value[turn].item(),
        )
        # print(encoded_frames.p2_empirical[turn])
        # print([read.policy_dim_labels[i] for i in encoded_frames.p2_choice_indices[turn]])

        return
    print(total)


import sys

if __name__ == "__main__":
    trajectories = read.read_build_trajectories("/home/user/train/data/0.build")

    i = int(sys.argv[1])
    for i in range(trajectories.size):

        if trajectories.score[i, 0] < 1:
            continue

        for index, p in zip(trajectories.actions[i], trajectories.policy[i]):
            pair = read.species_move_list[index]
            print(p, read.species_names[pair[0]], read.move_names[pair[1]])
        print(trajectories.eval[i])
        print(trajectories.score[i])

    # print(sum(trajectories.score) / trajectories.size)

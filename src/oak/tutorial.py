import sys

import argparse

parser = argparse.ArgumentParser(description="Oak Tutorial")
parser.add_argument("--main", required=True, type=str)
parser.add_argument("--data-path", default=None, type=str)
parser.add_argument("--network", default=None, type=str)


def test_consistency():
    fn_parser = parser.add_argument_group("")
    fn_parser.add_argument("--games", default=None, type=int)
    args = parser.parse_args()
    assert args.data_path, "Provide path to data file to inspect"
    assert args.network, "Provide path to network to test"

    import torch
    import oak
    import oak.torch


    network = oak.torch.BattleNetwork()

    with open(args.network, "rb") as f:
        network.read_parameters(f)

    buffer_list = oak.read_battle_data(args.data_path)

    max_games = min(args.games or len(buffer_list), len(buffer_list))

    for buffer, n_frames in buffer_list[: max_games]:
        encoded_frames = oak.EncodedBattleFrames.from_bytes(buffer, n_frames)
        encoded_frames_torch = oak.torch.EncodedBattleFrames(encoded_frames)
        output = oak.OutputBuffer(encoded_frames.size)
        output_torch = oak.torch.OutputBuffer(output)
        network.inference(encoded_frames_torch, output_torch)

        output_2 = oak.cpp_inference(
            args.network, oak.BattleFrames.from_bytes(buffer, n_frames)
        )
        output_torch_2 = oak.torch.OutputBuffer(output_2)

        assert torch.all(output_2.value == output_torch_2.value)
        assert torch.all(output_2.policy == output_torch_2.policy)

    print(f"CPP and Python agree on value/policy inference for the first {max_games} games of the data file.")

def battle_frame_stats():
    fn_parser = parser.add_argument_group("")
    args = parser.parse_args()
    assert args.data_path, "Provide path to recursively search for .battle.data files"

    files = oak.util.find_data_files(args.data_path, ext=".battle.data")
    print(f"Found {len(files)} data files")

    total_frames = 0
    total_battles = 0

    for file in files:
        data = oak.read_battle_data(file)
        for buf, n in data:
            total_battles += 1
            total_frames += n

    print(f"Total battle frames: {total_frames}")
    print(f"Average battle length: {total_frames / total_battles}")


def build_trajectory_stats():
    fn_parser = parser.add_argument_group("")
    args = parser.parse_args()

    # using only the head gives most recent files
    files = oak.util.find_data_files(".", ext=".build.data")
    assert len(files) > 0, "No build files found in cwd"

    from random import sample

    build_trajectories, read = oak.read_build_trajectories(files, 1024, 1)
    print(build_trajectories.size, read)
    assert build_trajectories.size == read, f"Bad read from {file}."
    for i in range(min(10, build_trajectories.size)):
        index = sample(list(range(build_trajectories.size)), 1)[0]
        print(f"Sample {index}:")
        species_move = [
            oak.species_move_list[_]
            for _ in build_trajectories.action[index].reshape(-1)
        ]
        names = []
        for sm in species_move:
            s, m = sm
            names.append(oak.species_names[s] + " " + oak.move_names[m])
        selection_probs = [
            int(1000 * float(_)) / 10
            for _ in build_trajectories.policy[index].reshape(-1)
        ]
        actions = build_trajectories.action[index].reshape(-1)

        data = [
            f"{n}:{p}" for n, p, a in zip(names, selection_probs, actions) if a >= 0
        ]

        l = 31
        print(data)
        print("value", build_trajectories.value[index].item())
        print("score", build_trajectories.score[index].item())
        # continue
        print("actions", build_trajectories.action[index, :l])
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


def create_team():
    fn_parser = parser.add_argument_group("")
    parser.add_argument("--max-pokemon", default=1, type=int)
    args = parser.parse_args()


    from oak.torch import BuildNetwork

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
        team = torch.zeros([len(oak.species_move_list)])

        # create mask for choosing the first species
        mask = torch.zeros([len(oak.species_move_list)])
        for index, pair in enumerate(oak.species_move_list):
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
        species, _ = oak.species_move_list[index]
        print(f"{oak.species_names[species]} : {p}% ~ {q}%")
        team[index] = 1

        # reset mask and fill with legal moves
        mask.zero_()
        n_moves = 0
        for index, pair in enumerate(oak.species_move_list):
            s, m = pair
            if s == species and m != 0:
                n_moves += 1
                mask[index] = 1

        for _ in range(min(4, n_moves)):
            index, p, q = sample_masked_logits(network.forward(team)[0], mask)
            _, move = oak.species_move_list[index]
            print(f"    {oak.move_names[move]} : {p}% ~ {q}%")
            team[index] = 1
            mask[index] = 0


def main():

    args = parser.parse_args()
    if args.main == "test-consistency":
        test_consistency()
    elif args.main == "battle-frame-stats":
        battle_frame_stats()
    elif args.main == "build-trajectory-stats":
        build_trajectory_stats()
    elif args.main == "create-team":
        create_team()
    else:
        assert False, "Bad --main arg"


if __name__ == "__main__":
    main()

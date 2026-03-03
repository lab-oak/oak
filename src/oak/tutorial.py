import sys

import argparse

parser = argparse.ArgumentParser(description="Oak Tutorial")
parser.add_argument("--main", default=None, type=str)
parser.add_argument("--data-path", default=None, type=str)
parser.add_argument("--network", default=None, type=str)
parser.add_argument("--build-network", default=None, type=str)


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

    for buffer, n_frames in buffer_list[:max_games]:
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

    print(
        f"CPP and Python agree on value/policy inference for the first {max_games} games of the data file."
    )


def battle_frame_stats():
    fn_parser = parser.add_argument_group("")
    args = parser.parse_args()
    assert args.data_path, "Provide path to recursively search for .battle.data files"

    import oak

    files = oak.util.find_data_files(args.data_path, ext=".battle.data")
    print(f"Found {len(files)} data files")

    total_frames = 0
    total_battles = 0
    iteration_counts = {}
    total_var = 0

    for file in files:
        data = oak.read_battle_data(file)

        for buf, n in data:
            frames = oak.BattleFrames.from_bytes(buf, n)

            for i in range(n):
                it = frames.iterations[i].item()
                if it in iteration_counts:
                    iteration_counts[it] += 1
                else:
                    iteration_counts[it] = 1

            total_battles += 1
            total_frames += n

    print(f"Total battle frames: {total_frames}")
    print(f"Average battle length: {total_frames / total_battles}")
    print(f"Iteration counts:")
    for key in iteration_counts:
        print(f"{key}: {iteration_counts[key]}")


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
    fn_parser.add_argument("--teams", default=1, type=int)
    fn_parser.add_argument("--max-pokemon", default=1, type=int)
    fn_parser.add_argument("--temp", default=1, type=float)
    fn_parser.add_argument("--max", default=1, type=float)
    args = parser.parse_args()

    import oak

    class Set:
        def __init__(self, species: int = 0):
            self.species = species
            self.moves = set()
            self.moves_remaining = set()
            if species:
                for i, sm in enumerate(oak.species_move_list):
                    s, m = sm
                    if s == self.species and m > 0:
                        self.moves_remaining.add(i)
                    if s > self.species:
                        break

        def add_move(self, index):
            self.moves.add(index)
            self.moves_remaining.remove(index)
            if len(moves) >= 4:
                self.moves_remaining = {}

        def print_(
            self,
        ):
            print(f"{self.species}: {[i for i in self.moves]}")

        def write_to_input(self, t):
            if self.species > 0:
                t[oak.species_move_table[self.species][0]] = 1.0
                for i in self.moves:
                    t[i] = 1.0

        def write_to_mask(self, t):
            t[oak.species_move_table[self.species][0]] = 0.0
            # requires adds?
            if len(self.moves_remaining):
                for i in self.moves_remaining:
                    t[i] = 1.0
            else:
                # clear all species, _ from mask
                for i, sm in enumerate(oak.species_move_list):
                    s, m = sm
                    if s == self.species:
                        t[i] = 0.0
                    if s > self.species:
                        break

    def apply_index_to_team(team, index):
        s, m = oak.species_move_list[index]
        for set_ in team:
            if s == set_.species:
                set_.add_move(index)

    if args.network:
        print("--network was provided but --build-network has priority")

    network_path = args.build_network or args.network or None

    if network_path:
        print(f"Using '{network_path}' as network path.")
    else:
        print("No build network path provided, using random network.")

    import torch
    from oak.torch import BuildNetwork

    network = BuildNetwork()
    if network_path:
        with open(network_path, "rb") as params:
            network.read_parameters(params)

    def sample_masked_logits(logits, mask):
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)
        probs_p = torch.softmax(masked_logits * args.temp, dim=-1)
        sampled = torch.multinomial(probs_p, 1)
        return (
            sampled,
            int(probs[sampled].item() * 10000) / 100,
            int(probs_p[sampled].item() * 10000) / 100,
        )

    def write_all_species_to_mask(mask, val):
        for i, sm in enumerate(oak.species_move_list):
            s, m = sm
            if m == 0:
                mask[i] = val

    for _ in range(args.teams):

        team = [Set() for _ in range(args.max_pokemon)]

        encoded_team = torch.zeros([len(oak.species_move_list)])
        mask = torch.zeros([len(oak.species_move_list)])
        # init mask with species
        write_all_species_to_mask(mask, 1.0)

        while (any(len(s.moves_remaining)) for s in team):

            for set_ in team:
                set_.write_to_input(encoded_team)

            # # create mask for choosing the first species
            # mask = torch.zeros([len(oak.species_move_list)])
            # for index, pair in enumerate(oak.species_move_list):
            #     s, m = pair
            #     if m == 0:
            #         mask[index] = 1

            # logits, _ = network.forward(team)
            # index, p, q = sample_masked_logits(logits, mask)
            # species, _ = oak.species_move_list[index]
            # print(f"{oak.species_names[species]} : {p}% ~ {q}%")
            # team[index] = 1

            # # reset mask and fill with legal moves
            # mask.zero_()
            # n_moves = 0
            # for index, pair in enumerate(oak.species_move_list):
            #     s, m = pair
            #     if s == species and m != 0:
            #         n_moves += 1
            #         mask[index] = 1

            # for _ in range(min(4, n_moves)):
            #     index, p, q = sample_masked_logits(network.forward(team)[0], mask)
            #     _, move = oak.species_move_list[index]
            #     print(f"    {oak.move_names[move]} : {p}% ~ {q}%")
            #     team[index] = 1
            #     mask[index] = 0

        team.print_()


def main():
    args = parser.parse_args()

    assert args.main

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

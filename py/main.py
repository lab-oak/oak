import sys

if len(sys.argv) < 3:
    print("Input: battle-file-path, buffer-size")
    exit()
import os

import torch

import libtrain
import net


def inference(
    network: net.Network, input: libtrain.EncodedFrameInput, output: net.OutputBuffers
):
    size = min(input.size, output.size)
    pokemon_output = network.pokemon_net.forward(input.pokemon[:size])
    active_output = network.active_net.forward(input.active[:size])
    pokemon_flat = torch.cat((input.hp[:size, :, 1:], pokemon_output), dim=3).view(
        -1, 2, 5 * (1 + network.pokemon_out_dim)
    )
    output.sides[:size, :, 0, 1 : network.active_out_dim + 1] = active_output[
        :, :, 0, :
    ]
    output.sides[:size, :, 0, 0] = input.hp[:size, :, 0, 0]
    output.sides[:size, :, 0, 1 + network.active_out_dim :] = pokemon_flat[:size]
    battle = output.sides[:size].view(size, 2 * network.side_out_dim)
    output.value[:size] = network.main_net.forward(battle)


def loss(input: libtrain.EncodedFrameInput, output: net.OutputBuffers):
    size = min(input.size, output.size)


class Optimizer:
    def __init__(self, network: torch.nn.Module):
        self.opt = torch.optim.Adam(network.parameters())


def find_battle_files(root_dir):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".battle"):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


def main():
    threads = 1

    parent = sys.argv[1]
    paths = find_battle_files(parent)
    print(f"{len(paths)} paths found")

    size = int(sys.argv[2])
    encoded_frames = libtrain.EncodedFrameInput(size)
    n_frames = libtrain.encode_buffers(
        paths, threads, size, encoded_frames, start_index=0, write_prob=0.2
    )
    print(f"read {n_frames} encoded frames")

    network = net.Network()
    output_buffer = net.OutputBuffers(size)
    inference(network, encoded_frames, output_buffer)

    opt = Optimizer(network)


if __name__ == "__main__":
    main()

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
    output.pokemon[:size] = network.pokemon_net.forward(input.pokemon[:size])
    output.active[:size] = network.active_net.forward(input.active[:size])
    # active hp
    output.sides[:size, :, :, 0] = input.hp[:size, :, :1, 0]
    # active word
    output.sides[:size, :, :, 1 : network.active_out_dim + 1] = output.active[
        :size, :, :, :
    ]
    # pokemon hp/word
    pokemon_flat = torch.cat(
        (input.hp[:size, :, 1:], output.pokemon[:size]), dim=3
    ).view(size, 2, 1, 5 * (1 + network.pokemon_out_dim))
    output.sides[:size, :, :, 1 + network.active_out_dim :] = pokemon_flat[:size]
    battle = output.sides[:size].view(size, 2 * network.side_out_dim)
    output.value[:size] = network.main_net.forward(battle)


def loss(input: libtrain.EncodedFrameInput, output: net.OutputBuffers):
    size = min(input.size, output.size)


class Optimizer:
    def __init__(self, network: torch.nn.Module):
        self.opt = torch.optim.Adam(network.parameters())

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


def find_battle_files(root_dir):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".battle"):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


def loss(input: libtrain.EncodedFrameInput, output: net.OutputBuffers):
    size = min(input.size, output.size)

    w_nash = 0.45
    w_empirical = 0.45
    w_score = 0.1

    target = (
        w_nash * input.nash_value[:size]
        + w_empirical * input.empirical_value[:size]
        + w_score * input.score[:size]
    )

    return torch.nn.functional.mse_loss(output.value[:size], target)


def main():
    threads = 1
    steps = 500

    parent = sys.argv[1]
    paths = find_battle_files(parent)
    print(f"{len(paths)} paths found")

    size = int(sys.argv[2])
    encoded_frames = libtrain.EncodedFrameInput(size)
    output_buffer = net.OutputBuffers(size)

    network = net.Network()
    with open("netparams", "wb") as f:
        network.write_parameters(f)

    optimizer = Optimizer(network)

    for step in range(steps):
        encoded_frames.clear()
        output_buffer.clear()

        n_frames = libtrain.encode_buffers(
            paths, threads, size, encoded_frames, start_index=0, write_prob=0.2
        )

        inference(network, encoded_frames, output_buffer)

        optimizer.zero_grad()
        loss_value = loss(encoded_frames, output_buffer)
        loss_value.backward()
        optimizer.step()

        print(f"Step {step+1}/{steps}, Loss: {loss_value.item():.6f}")

    with open("netparams2", "wb") as f:
        network.write_parameters(f)

if __name__ == "__main__":
    main()

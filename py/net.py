import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import libtrain
import struct


class Affine(nn.Module):
    def __init__(self, in_dim, out_dim, clamp=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.clamp = clamp

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        if self.clamp:
            return torch.clamp(x, 0.0, 1.0)
        return x

    def read_parameters(self, f):
        dims = f.read(8)
        in_dim, out_dim = struct.unpack("<II", dims)
        assert in_dim == self.in_dim, f"Expected in_dim={self.in_dim}, got {in_dim}"
        assert (
            out_dim == self.out_dim
        ), f"Expected out_dim={self.out_dim}, got {out_dim}"
        self.weight.data.copy_(
            torch.frombuffer(
                f.read(self.weight.numel() * 4), dtype=torch.float32
            ).reshape(self.weight.shape)
        )
        self.bias.data.copy_(
            torch.frombuffer(f.read(self.bias.numel() * 4), dtype=torch.float32)
        )

    def write_parameters(self, f):
        f.write(struct.pack("<II", self.in_dim, self.out_dim))
        f.write(self.weight.detach().cpu().numpy().astype("f4").tobytes())
        f.write(self.bias.detach().cpu().numpy().astype("f4").tobytes())


class EmbeddingNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, clamp0=True, clamp1=True):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim, clamp=clamp0)
        self.fc1 = Affine(hidden_dim, out_dim, clamp=clamp1)

    def forward(self, x):
        return self.fc1(self.fc0(x))

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.fc1.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.fc1.write_parameters(f)


class MainNet(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, value_hidden_dim, policy_hidden_dim, policy_out_dim
    ):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim)
        self.value_fc1 = Affine(hidden_dim, value_hidden_dim)
        self.value_fc2 = Affine(value_hidden_dim, 1, clamp=False)
        self.policy1_fc1 = Affine(hidden_dim, policy_hidden_dim)
        self.policy1_fc2 = Affine(policy_hidden_dim, policy_out_dim, clamp=False)
        self.policy2_fc1 = Affine(hidden_dim, policy_hidden_dim)
        self.policy2_fc2 = Affine(policy_hidden_dim, policy_out_dim, clamp=False)

    def forward(self, x):
        b0 = self.fc0(x)
        value_b1 = self.value_fc1(b0)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)
        return value

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.value_fc1.read_parameters(f)
        self.value_fc2.read_parameters(f)
        self.policy1_fc1.read_parameters(f)
        self.policy1_fc2.read_parameters(f)
        self.policy2_fc1.read_parameters(f)
        self.policy2_fc2.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.value_fc1.write_parameters(f)
        self.value_fc2.write_parameters(f)
        self.policy1_fc1.write_parameters(f)
        self.policy1_fc2.write_parameters(f)
        self.policy2_fc1.write_parameters(f)
        self.policy2_fc2.write_parameters(f)


class Network(torch.nn.Module):

    hidden_dim = libtrain.hidden_dim
    pokemon_hidden_dim = libtrain.value_hidden_dim
    pokemon_out_dim = libtrain.pokemon_out_dim
    active_hidden_dim = libtrain.active_hidden_dim
    active_out_dim = libtrain.active_out_dim
    side_out_dim = libtrain.side_out_dim
    hidden_dim = libtrain.hidden_dim
    value_hidden_dim = libtrain.value_hidden_dim
    policy_hidden_dim = libtrain.policy_hidden_dim
    policy_out_dim = libtrain.policy_out_dim
    pokemon_in_dim = libtrain.pokemon_in_dim
    active_in_dim = libtrain.active_in_dim

    def __init__(self):
        super().__init__()
        self.pokemon_net = EmbeddingNet(
            self.pokemon_in_dim, self.pokemon_hidden_dim, self.pokemon_out_dim
        )
        self.active_net = EmbeddingNet(
            self.active_in_dim, self.active_hidden_dim, self.active_out_dim
        )
        self.main_net = MainNet(
            self.hidden_dim,
            self.hidden_dim,
            self.value_hidden_dim,
            self.policy_hidden_dim,
        )

    def read_parameters(self, f):
        self.pokemon_net.read_parameters(f)
        self.active_net.read_parameters(f)
        self.main_net.read_parameters(f)

    def write_parameters(self, f):
        self.pokemon_net.write_parameters(f)
        self.active_net.write_parameters(f)
        self.main_net.write_parameters(f)


class OutputBuffers:
    def __init__(self, size):
        self.pokemon = np.array((size, 2, 5, Network.pokemon_out_dim), dtype=np.float32)
        self.active = np.array((size, 2, 1, Network.active_out_dim), dtype=np.float32)
        self.sides = np.array(
            (size, 2, (1 + Network.active_out_dim) + 5 * (1 + Network.pokemon_out_dim)),
            dtype=np.float32,
        )
        self.value = np.array((size, 1), dtype=np.float32)


def inference(
    self, network: Network, input: libtrain.EncodedFrameInput, output: OutputBuffers
):
    pokemon_output = network.pokemon_net.forward(input.pokemon)
    active_output = network.active_net.forward(input.active)
    for s in range(2):
        self.sides[:, s, 1 : Network.active_out_dim + 1] = active_output[:, s, 0, :]
        self.sides[:, s, 0] = input.hp[:, s, 0]
    self.sides[:, 1, :] = active_output[:, 1, 0, :]
    main_input = torch.cat(
        (
            active_output.view(-1, 2, 5 * Network.pokemon_out_dim),
            pokemon_output.view(-1, 2, 5 * Network.pokemon_out_dim),
        ),
        dim=2,
    ).view(-1, 1 + Network.active_out_dim + 5 * (1 + Network.pokemon_out_dim))
    output.value = network.main_net.forward(main_input)

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyoak

import struct

import torch


class BuildTrajectoryTorch:
    def __init__(self, traj: pyoak.BuildTrajectory, n=None, device="cpu"):
        if n is None:
            n = traj.size
        self.size = n
        self.actions = torch.from_numpy(traj.actions[:, :n]).long().to(device)
        self.mask = torch.from_numpy(traj.mask[:, :n]).long().to(device)
        self.policy = torch.from_numpy(traj.policy[:, :n]).float().to(device)
        self.value = torch.from_numpy(traj.value[:, :n]).float().to(device)
        self.score = torch.from_numpy(traj.score[:, :n]).float().to(device)
        self.start = torch.from_numpy(traj.start[:, :n]).long().to(device)
        self.end = torch.from_numpy(traj.end[:, :n]).long().to(device)


class BattleFrameTorch:
    def __init__(self, encoded_frame: pyoak.EncodedBattleFrame):
        self.size = encoded_frame.size
        self.m = torch.from_numpy(encoded_frame.m)
        self.n = torch.from_numpy(encoded_frame.n)
        self.p1_choice_indices = torch.from_numpy(encoded_frame.p1_choice_indices)
        self.p2_choice_indices = torch.from_numpy(encoded_frame.p2_choice_indices)
        self.pokemon = torch.from_numpy(encoded_frame.pokemon)
        self.active = torch.from_numpy(encoded_frame.active)
        self.hp = torch.from_numpy(encoded_frame.hp)
        self.p1_empirical = torch.from_numpy(encoded_frame.p1_empirical)
        self.p1_nash = torch.from_numpy(encoded_frame.p1_nash)
        self.p2_empirical = torch.from_numpy(encoded_frame.p2_empirical)
        self.p2_nash = torch.from_numpy(encoded_frame.p2_nash)
        self.empirical_value = torch.from_numpy(encoded_frame.empirical_value)
        self.nash_value = torch.from_numpy(encoded_frame.nash_value)
        self.score = torch.from_numpy(encoded_frame.score)

    def permute_pokemon(self):
        perms = torch.stack([torch.randperm(5) for _ in range(self.size)], dim=0)
        perms_expanded = perms[:, None, :, None].expand(-1, 2, -1, pyoak.pokemon_in_dim)
        torch.gather(self.pokemon, dim=2, index=perms_expanded)

    def permute_sides(self, prob=0.5):
        mask = torch.rand(self.size) < prob

        self.pokemon[mask] = self.pokemon[mask].flip(dims=[1])
        self.active[mask] = self.active[mask].flip(dims=[1])
        self.hp[mask] = self.hp[mask].flip(dims=[1])

        self.empirical_value[mask] = 1 - self.empirical_value[mask]
        self.nash_value[mask] = 1 - self.nash_value[mask]
        self.score[mask] = 1 - self.score[mask]

        def swap(mask, x, y):
            temp = x[mask].clone().detach()
            x[mask] = y[mask]
            y[mask] = temp

        swap(mask, self.m, self.n)
        swap(mask, self.p1_choice_indices, self.p2_choice_indices)
        swap(mask, self.p1_empirical, self.p2_empirical)
        swap(mask, self.p1_nash, self.p2_nash)


class Affine(nn.Module):
    def __init__(self, in_dim, out_dim, clamp=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = torch.nn.Linear(in_dim, out_dim)
        self.clamp = clamp

    # Read the parameters in the same way as the C++ net
    def read_parameters(self, f):
        dims = f.read(8)
        in_dim, out_dim = struct.unpack("<II", dims)
        assert in_dim == self.in_dim, f"Expected in_dim={self.in_dim}, got {in_dim}"
        assert (
            out_dim == self.out_dim
        ), f"Expected out_dim={self.out_dim}, got {out_dim}"
        self.layer.bias.data.copy_(
            torch.frombuffer(
                bytearray(f.read(self.layer.bias.numel() * 4)), dtype=torch.float32
            )
        )
        self.layer.weight.data.copy_(
            torch.frombuffer(
                bytearray(f.read(self.layer.weight.numel() * 4)), dtype=torch.float32
            ).reshape(self.layer.weight.shape)
        )

    def write_parameters(self, f):
        f.write(struct.pack("<II", self.in_dim, self.out_dim))
        f.write(self.layer.bias.detach().cpu().numpy().astype("f4").tobytes())
        f.write(self.layer.weight.detach().cpu().numpy().astype("f4").tobytes())

    # To make the network quantizable using the Stockfish scheme
    def clamp_parameters(self):
        self.layer.weight.data.clamp_(-2, 2)

    def forward(self, x):
        x = self.layer(x)
        if self.clamp:
            return torch.clamp(x, 0.0, 1.0)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, clamp0=True, clamp1=True):
        super().__init__()
        self.fc0 = Affine(in_dim, hidden_dim, clamp=clamp0)
        self.fc1 = Affine(hidden_dim, out_dim, clamp=clamp1)

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.fc1.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.fc1.write_parameters(f)

    def clamp_parameters(
        self,
    ):
        self.fc0.clamp_parameters()
        self.fc1.clamp_parameters()

    def forward(self, x):
        return self.fc1(self.fc0(x))


class BuildNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = EmbeddingNet(
            pyoak.species_move_list_size,
            pyoak.build_policy_hidden_dim,
            pyoak.species_move_list_size,
            True,
            False,
        )
        self.value_net = EmbeddingNet(
            pyoak.species_move_list_size, pyoak.build_value_hidden_dim, 1, True, False
        )

    def read_parameters(self, f):
        self.policy_net.read_parameters(f)
        self.value_net.read_parameters(f)

    def write_parameters(self, f):
        self.policy_net.write_parameters(f)
        self.value_net.write_parameters(f)

    def forward(self, x):
        return self.policy_net.forward(x), self.value_net.forward(x)


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

    def clamp_parameters(
        self,
    ):
        self.fc0.clamp_parameters()
        self.value_fc1.clamp_parameters()
        self.value_fc2.clamp_parameters()
        self.policy1_fc1.clamp_parameters()
        self.policy1_fc2.clamp_parameters()
        self.policy2_fc1.clamp_parameters()
        self.policy2_fc2.clamp_parameters()

    def forward(self, x):
        b0 = self.fc0(x)
        value_b1 = self.value_fc1(b0)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)
        p1_policy_b1 = self.policy1_fc1(b0)
        p1_policy_b2 = self.policy1_fc2(p1_policy_b1)
        p2_policy_b1 = self.policy2_fc1(b0)
        p2_policy_b2 = self.policy2_fc2(p2_policy_b1)
        return value, p1_policy_b2, p2_policy_b2

    def forward_value_only(self, x):
        b0 = self.fc0(x)
        value_b1 = self.value_fc1(b0)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)
        return value


# holds the output of the embedding nets, the input to main net, and value/policy output of main net
class OutputBuffers:
    def __init__(self, size):
        self.size = size
        self.pokemon = torch.zeros(
            (size, 2, 5, BattleNetwork.pokemon_out_dim), dtype=torch.float32
        )
        self.active = torch.zeros(
            (size, 2, 1, BattleNetwork.active_out_dim), dtype=torch.float32
        )
        self.sides = torch.zeros((size, 2, 1, 256), dtype=torch.float32)
        self.value = torch.zeros((size, 1), dtype=torch.float32)
        # last dim in neg inf to supply logit for invalid actions
        self.p1_policy_logit = torch.zeros(size, pyoak.policy_out_dim + 1)
        self.p2_policy_logit = torch.zeros(size, pyoak.policy_out_dim + 1)
        self.p1_policy = torch.zeros((size, 9))
        self.p2_policy = torch.zeros((size, 9))

    def clear(self):
        self.pokemon.detach_().zero_()
        self.active.detach_().zero_()
        self.sides.detach_().zero_()
        self.value.detach_().zero_()
        self.p1_policy_logit.detach_().zero_()
        self.p2_policy_logit.detach_().zero_()
        self.p1_policy_logit[:, -1] = -torch.inf
        self.p2_policy_logit[:, -1] = -torch.inf
        self.p1_policy.detach_().zero_()
        self.p2_policy.detach_().zero_()


class BattleNetwork(torch.nn.Module):
    pokemon_hidden_dim = pyoak.pokemon_hidden_dim
    pokemon_out_dim = pyoak.pokemon_out_dim
    active_hidden_dim = pyoak.active_hidden_dim
    active_out_dim = pyoak.active_out_dim
    side_out_dim = pyoak.side_out_dim
    hidden_dim = pyoak.hidden_dim
    value_hidden_dim = pyoak.value_hidden_dim
    policy_hidden_dim = pyoak.policy_hidden_dim
    policy_out_dim = pyoak.policy_out_dim
    pokemon_in_dim = pyoak.pokemon_in_dim
    active_in_dim = pyoak.active_in_dim

    def __init__(self):
        super().__init__()
        self.pokemon_net = EmbeddingNet(
            self.pokemon_in_dim, self.pokemon_hidden_dim, self.pokemon_out_dim
        )
        self.active_net = EmbeddingNet(
            self.active_in_dim, self.active_hidden_dim, self.active_out_dim
        )
        self.main_net = MainNet(
            2 * self.side_out_dim,
            self.hidden_dim,
            self.value_hidden_dim,
            self.policy_hidden_dim,
            self.policy_out_dim,
        )

    def read_parameters(self, f):
        self.pokemon_net.read_parameters(f)
        self.active_net.read_parameters(f)
        self.main_net.read_parameters(f)

    def write_parameters(self, f):
        self.pokemon_net.write_parameters(f)
        self.active_net.write_parameters(f)
        self.main_net.write_parameters(f)

    def clamp_parameters(self):
        self.pokemon_net.clamp_parameters()
        self.active_net.clamp_parameters()
        self.main_net.clamp_parameters()

    def inference(self, input: BattleFrameTorch, output: OutputBuffers):
        size = min(input.size, output.size)
        output.pokemon[:size] = self.pokemon_net.forward(input.pokemon[:size])
        output.active[:size] = self.active_net.forward(input.active[:size])
        # mask output for hp
        output.pokemon[:size] *= (input.hp[:size, :, 1:] != 0).float()
        output.active[:size] *= (input.hp[:size, :, :1] != 0).float()
        # active hp
        output.sides[:size, :, :, 0] = input.hp[:size, :, :1, 0]
        # active word
        output.sides[:size, :, :, 1 : self.active_out_dim + 1] = output.active[:size]
        # pokemon hp/word
        pokemon_flat = torch.cat(
            (input.hp[:size, :, 1:], output.pokemon[:size]), dim=3
        ).view(size, 2, 1, 5 * (1 + self.pokemon_out_dim))
        output.sides[:size, :, :, 1 + self.active_out_dim :] = pokemon_flat[:size]
        battle = output.sides[:size].view(size, 2 * self.side_out_dim)
        (
            output.value[:size],
            output.p1_policy_logit[:size, :-1],
            output.p2_policy_logit[:size, :-1],
        ) = self.main_net.forward(battle)
        output.p1_policy[:size] = torch.gather(
            output.p1_policy_logit, 1, input.p1_choice_indices[:size]
        )
        output.p2_policy[:size] = torch.gather(
            output.p2_policy_logit, 1, input.p2_choice_indices[:size]
        )

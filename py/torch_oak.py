import torch
import torch.nn as nn
import torch.nn.functional as F

import py_oak

import struct
import hashlib
from typing import Dict, List

import torch


class BattleFrameSampler:
    def __init__(self, paths):
        self.paths = paths
        self.path_frame_counts: List[int] = []
        self.path_frame_data: List[List[[int, int]]] = []

        # for path in paths
        #     data: List[[bytes, int]] = read_battle_data(path)
        #     path_frame_counts.append(sum(n for _, n in data))


class EncodedBattleFrames:
    def __init__(self, frames: py_oak.EncodedBattleFrames):
        self.size = frames.size
        self.m = torch.from_numpy(frames.m)
        self.n = torch.from_numpy(frames.n)
        self.p1_choice_indices = torch.from_numpy(frames.p1_choice_indices)
        self.p2_choice_indices = torch.from_numpy(frames.p2_choice_indices)
        self.pokemon = torch.from_numpy(frames.pokemon)
        self.active = torch.from_numpy(frames.active)
        self.hp = torch.from_numpy(frames.hp)
        self.p1_empirical = torch.from_numpy(frames.p1_empirical)
        self.p1_nash = torch.from_numpy(frames.p1_nash)
        self.p2_empirical = torch.from_numpy(frames.p2_empirical)
        self.p2_nash = torch.from_numpy(frames.p2_nash)
        self.empirical_value = torch.from_numpy(frames.empirical_value)
        self.nash_value = torch.from_numpy(frames.nash_value)
        self.score = torch.from_numpy(frames.score)

    def permute_pokemon(self):
        perms = torch.stack([torch.randperm(5) for _ in range(self.size)], dim=0)
        perms_expanded = perms[:, None, :, None].expand(
            -1, 2, -1, py_oak.pokemon_in_dim
        )
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


def hash_bytes(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def combine_hash(h1: int, h2: int) -> int:
    # A simple 64-bit mixing function
    return (h1 ^ (h2 + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2))) & 0xFFFFFFFFFFFFFFFF


class BuildTrajectories:
    def __init__(self, traj: py_oak.BuildTrajectories, n=None, device="cpu"):
        if n is None:
            n = 31
        self.size = traj.size
        self.actions = torch.from_numpy(traj.actions[:, :n]).long().to(device)
        self.mask = torch.from_numpy(traj.mask[:, :n]).long().to(device)
        self.policy = torch.from_numpy(traj.policy[:, :n]).float().to(device)
        self.value = torch.from_numpy(traj.value[:, :n]).float().to(device)
        self.score = torch.from_numpy(traj.score[:, :n]).float().to(device)
        self.start = torch.from_numpy(traj.start[:, :n]).long().to(device)
        self.end = torch.from_numpy(traj.end[:, :n]).long().to(device)

    def sample(self, p=1):
        r = torch.rand((self.size,)) < p
        with torch.no_grad():
            self.actions = self.actions[r].clone()
            self.mask = self.mask[r].clone()
            self.policy = self.policy[r].clone()
            self.value = self.value[r].clone()
            self.score = self.score[r].clone()
            self.start = self.start[r].clone()
            self.end = self.end[r].clone()
        self.size = sum(r).item()


# Networks


class Affine(nn.Module):
    def __init__(self, in_dim, out_dim, clamp=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = torch.nn.Linear(in_dim, out_dim)
        self.clamp = clamp

    def read_parameters(self, f):
        dims = f.read(8)
        in_dim, out_dim = struct.unpack("<II", dims)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = torch.nn.Linear(self.in_dim, self.out_dim)

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

    def clamp_parameters(self):
        self.layer.weight.data.clamp_(-2, 2)

    def forward(self, x):
        x = self.layer(x)
        if self.clamp:
            return torch.clamp(x, 0.0, 1.0)
        return x

    def hash(self) -> int:
        data = (
            self.layer.weight.detach().cpu().numpy().astype("f4").tobytes()
            + self.layer.bias.detach().cpu().numpy().astype("f4").tobytes()
            + struct.pack("<?II", self.clamp, self.in_dim, self.out_dim)
        )
        return hash_bytes(data)


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

    def clamp_parameters(self):
        self.fc0.clamp_parameters()
        self.fc1.clamp_parameters()

    def forward(self, x):
        return self.fc1(self.fc0(x))

    def hash(self) -> int:
        h = self.fc0.hash()
        h = combine_hash(h, self.fc1.hash())
        return h


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

    def clamp_parameters(self):
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

    def hash(self) -> int:
        h = self.fc0.hash()
        for sub in [
            self.value_fc1,
            self.value_fc2,
            self.policy1_fc1,
            self.policy1_fc2,
            self.policy2_fc1,
            self.policy2_fc2,
        ]:
            h = combine_hash(h, sub.hash())
        return h


# holds the output of the embedding nets, the input to main net, and value/policy output of main net
class OutputBuffers:
    def __init__(self, size, pod=py_oak.pokemon_out_dim, aod=py_oak.active_out_dim):
        self.size = size
        self.pokemon_out_dim = pod
        self.active_out_dim = aod
        self.side_out_dim = aod + 5 * pod
        self.pokemon = torch.zeros(
            (size, 2, 5, self.pokemon_out_dim), dtype=torch.float32
        )
        self.active = torch.zeros(
            (size, 2, 1, self.active_out_dim), dtype=torch.float32
        )
        self.sides = torch.zeros((size, 2, 1, self.side_out_dim), dtype=torch.float32)
        self.value = torch.zeros((size, 1), dtype=torch.float32)
        # last dim is neg inf to supply logit for invalid actions
        self.p1_policy_logit = torch.zeros(size, py_oak.policy_out_dim + 1)
        self.p2_policy_logit = torch.zeros(size, py_oak.policy_out_dim + 1)
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
    # only remaining hard-coded dims
    pokemon_in_dim = py_oak.pokemon_in_dim
    active_in_dim = py_oak.active_in_dim
    policy_out_dim = py_oak.policy_out_dim

    def __init__(
        self,
        phd=py_oak.pokemon_hidden_dim,
        ahd=py_oak.active_hidden_dim,
        pod=py_oak.pokemon_out_dim,
        aod=py_oak.active_out_dim,
        hd=py_oak.hidden_dim,
        vhd=py_oak.value_hidden_dim,
        pohd=py_oak.policy_hidden_dim,
    ):
        super().__init__()
        self.pokemon_hidden_dim = phd
        self.active_hidden_dim = ahd
        self.pokemon_out_dim = pod
        self.active_out_dim = aod
        self.side_out_dim = aod + 5 * pod
        self.hidden_dim = hd
        self.value_hidden_dim = vhd
        self.policy_hidden_dim = pohd

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

    def inference(
        self, input: EncodedBattleFrames, output: OutputBuffers, use_policy: bool = True
    ):
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

        if use_policy:
            (
                output.value[:size],
                output.p1_policy_logit[:size, :-1],
                output.p2_policy_logit[:size, :-1],
            ) = self.main_net.forward(battle)
        else:
            output.value = self.main_net.forward_value_only(battle)

        output.p1_policy[:size] = torch.gather(
            output.p1_policy_logit, 1, input.p1_choice_indices[:size]
        )
        output.p2_policy[:size] = torch.gather(
            output.p2_policy_logit, 1, input.p2_choice_indices[:size]
        )

    def hash(self) -> int:
        h = self.pokemon_net.hash()
        h = combine_hash(h, self.active_net.hash())
        h = combine_hash(h, self.main_net.hash())
        return h & 0xFFFFFFFFFFFFFFFF


class BuildNetwork(nn.Module):
    def __init__(
        self,
        policy_hidden_dim=py_oak.build_policy_hidden_dim,
        value_hidden_dim=py_oak.build_value_hidden_dim,
    ):
        super().__init__()
        self.policy_net = EmbeddingNet(
            py_oak.species_move_list_size,
            policy_hidden_dim,
            py_oak.species_move_list_size,
            True,
            False,
        )
        self.value_net = EmbeddingNet(
            py_oak.species_move_list_size, value_hidden_dim, 1, True, False
        )

    def read_parameters(self, f):
        self.policy_net.read_parameters(f)
        self.value_net.read_parameters(f)

    def write_parameters(self, f):
        self.policy_net.write_parameters(f)
        self.value_net.write_parameters(f)

    def forward(self, x):
        return self.policy_net.forward(x), self.value_net.forward(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        in_dim, out_dim = struct.unpack('<II', dims)
        assert in_dim == self.in_dim, f"Expected in_dim={self.in_dim}, got {in_dim}"
        assert out_dim == self.out_dim, f"Expected out_dim={self.out_dim}, got {out_dim}"

        self.weight.data.copy_(
            torch.frombuffer(f.read(self.weight.numel() * 4), dtype=torch.float32)
            .reshape(self.weight.shape)
        )
        self.bias.data.copy_(
            torch.frombuffer(f.read(self.bias.numel() * 4), dtype=torch.float32)
        )

    def write_parameters(self, f):
        f.write(struct.pack('<II', self.in_dim, self.out_dim))
        f.write(self.weight.detach().cpu().numpy().astype('f4').tobytes())
        f.write(self.bias.detach().cpu().numpy().astype('f4').tobytes())

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
    def __init__(self, hidden_dim, policy_out_dim):
        super().__init__()
        in_dim = 512
        self.fc0 = Affine(in_dim, hidden_dim)
        self.value_fc1 = Affine(hidden_dim, hidden_dim)
        self.value_fc2 = Affine(hidden_dim, 1, clamp=False)
        self.policy_fc1 = Affine(hidden_dim, hidden_dim)
        self.policy_fc2 = Affine(hidden_dim, policy_out_dim, clamp=False)

    def forward(self, x, return_policy=True):
        b0 = self.fc0(x)
        value_b1 = self.value_fc1(b0)
        value_b2 = self.value_fc2(value_b1)
        value = torch.sigmoid(value_b2)

        policy_output = None
        if return_policy:
            policy_b1 = self.policy_fc1(b0)
            policy_output = self.policy_fc2(policy_b1)
        return value, policy_output

    def read_parameters(self, f):
        self.fc0.read_parameters(f)
        self.value_fc1.read_parameters(f)
        self.value_fc2.read_parameters(f)

    def write_parameters(self, f):
        self.fc0.write_parameters(f)
        self.value_fc1.write_parameters(f)
        self.value_fc2.write_parameters(f)

class Network(torch.nn.Module):
    pokemon_hidden_dim = 32
    pokemon_out_dim = 39
    active_hidden_dim = 32
    active_out_dim = 55
    side_out_dim = 256
    main_hidden_dim = 64
    policy_dim = 151 + 164

    def __init__(self, pokemon_in_dim, active_in_dim):
        super().__init__()
        self.pokemon_in_dim = pokemon_in_dim
        self.active_in_dim = active_in_dim
        self.p = EmbeddingNet(pokemon_in_dim, self.pokemon_hidden_dim, self.pokemon_out_dim)
        self.a = EmbeddingNet(active_in_dim, self.active_hidden_dim, self.active_out_dim)
        self.m = MainNet(self.main_hidden_dim, self.policy_dim)

    def read_parameters(self, f):
        self.p.read_parameters(f)
        self.a.read_parameters(f)
        self.m.read_parameters(f)

    def write_parameters(self, f):
        self.p.write_parameters(f)
        self.a.write_parameters(f)
        self.m.write_parameters(f)

    def forward(self, pokemon_input, active_input):
        batch_size = pokemon_input.shape[0]
        assert pokemon_input.shape == (batch_size, 2, 5, self.pokemon_in_dim)
        assert active_input.shape == (batch_size, 2, 1, self.active_in_dim)

        pokemon_out = self.p.forward(pokemon_input.view(-1, self.pokemon_in_dim))
        pokemon_out = pokemon_out.view(batch_size, 2, 5, self.pokemon_out_dim)
        active_out = self.a.forward(active_input.view(-1, self.active_in_dim))
        active_out = active_out.view(batch_size, 2, 1, self.active_out_dim)

        side_1 = torch.cat([
            active_out[:, 0].view(batch_size, -1),    # (batch_size, active_out_dim)
            pokemon_out[:, 0].view(batch_size, -1)    # (batch_size, 5 * pokemon_out_dim)
        ], dim=1)  # (batch_size, side_out_dim)

        side_2 = torch.cat([
            active_out[:, 1].view(batch_size, -1),
            pokemon_out[:, 1].view(batch_size, -1)
        ], dim=1)

        main_input = torch.cat([side_1, side_2], dim=1)  # (batch_size, 2 * side_out_dim)
        print(main_input.shape)
        assert main_input.shape == (batch_size, 512)

        value = self.m.forward(main_input)  # (batch_size, out_dim)
        return value

def main():

    pokemon_in_dim = 198
    active_in_dim = 428

    net = Network(pokemon_in_dim, active_in_dim)

    with open("./testsave", "rb") as f:
        net.read_parameters(f)

    with torch.no_grad():
        batch_size = 1
        pokemon_input = torch.zeros((batch_size, 2, 5, pokemon_in_dim))
        active_input = torch.zeros((batch_size, 2, 1, active_in_dim))
        out = net.forward(pokemon_input, active_input)

        print(out)

if __name__ == "__main__":
    main()
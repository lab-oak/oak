import torch
import torch.nn as nn
import torch.nn.functional as F

import libtrain
import net



class SharedBuffers:
    def __init__(self, batch_size):
        self.pokemon_buffer = mp.Array('f', batch_size * 2 * 5 * net.Network.pokemon_in_dim, lock=False)
        self.active_buffer = mp.Array('f', batch_size * 2 * 1 * net.Network.active_in_dim, lock=False)
        self.score_buffer = mp.Array('f', batch_size * 1, lock=False)
        self.eval_buffer = mp.Array('f', batch_size * 1, lock=False)
        self.acc_buffer = mp.Array('f', batch_size * 512, lock = False)

    def to_tensor(self, i = None):
        if i is None:
            return [
                torch.frombuffer(self.pokemon_buffer, dtype=torch.float).view(-1, 2, 5, net.Network.pokemon_in_dim,),
                torch.frombuffer(self.active_buffer, dtype=torch.float).view(-1, 2, 1, net.Network.active_in_dim,),
                torch.frombuffer(self.score_buffer, dtype=torch.float).view(-1, 1),
                torch.frombuffer(self.eval_buffer, dtype=torch.float).view(-1, 1),
                torch.frombuffer(self.acc_buffer, dtype=torch.float).view(-1, 2, 1, 256)]
        p, a, s, e, acc = self.to_tensor()
        return p[i], a[i], s[i], e[i], acc[i]

    def forward(self, pokemon_net, active_net, main_net, print_buffer=False):
        p, a, s, e, acc = self.to_tensor() 
        pokemon_out = pokemon_net.forward(p, print_buffer)
        active_out = active_net.forward(a, print_buffer)

        # write words to acc layer, offset by 1 for the hp entry
        for player in range(2):
            a_ = active_out[:, player, 0]
            assert(a_.shape[1] == frame.ACTIVE_OUT - 1)
            acc[:, player, 0, 1 : frame.ACTIVE_OUT] = a_
            for _ in range(5):
                start_index = frame.ACTIVE_OUT + _ * frame.POKEMON_OUT
                acc[:, player, 0, (start_index + 1) : start_index + frame.POKEMON_OUT] = pokemon_out[:, player, _]

        main_net_input = torch.concat([acc[:, 0, 0, :], acc[:, 1, 0, :]], dim=1)
        value_out, policy_out = main_net.forward(main_net_input, print_buffer)
        return torch.sigmoid(value_out)

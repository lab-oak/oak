#pragma once

#include <encode/battle-key.h>
#include <encode/battle.h>

template <int dim = 39> struct PokemonCache {

  using Slot = std::array<std::array<float, dim>, 256>;

  using Side = std::array<Slot, 6>;
  using Battle = std::array<Side, 2>;

  Battle data;

  static void fill_slot(const auto &net, auto &slot,
                        const PKMN::Pokemon &pokemon) {
    for (auto si = 0; si <= Encode::Status::n_dim; ++si) {
      for (uint8_t b = 0; b < 16; ++b) {
        auto copy = pokemon;
        for (auto m = 0; m < 4; ++m) {
          copy.moves.pp = b & (1 << m);
        }
        if (si < Encode::Status::n_dim) {
          copy.status = static_cast<uint8_t>()
        }
        std::array<float, Encode::Pokemon::n_dim> input;
        Encode::Pokemon::write(copy, 0, input.data());
        if (si < Encode::Status::n_dim) {
          input[100 + si] = 1;
        } // == n_dim means no status
        auto key = Encode::pokemon_key(copy);
        net.propagate(input.data(), slot[])
      }
    }
  }

  PokemonCache(const PKMN::Battle &battle) {
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        fill_slot(data[s][p], battle.sides[s].pokemon[p]);
      }
    }
  }
}
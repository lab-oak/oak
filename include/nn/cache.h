#pragma once

#include <encode/battle-key.h>
#include <encode/battle.h>
#include <encode/team.h>

namespace NN {

template <int dim = 39> struct PokemonCache {

  static constexpr auto n_status = Encode::Status::n_dim + 1;
  static constexpr auto n_pp = 16;
  static constexpr uint8_t n_embeddings = n_status * n_pp;

  using Slot = std::array<std::array<float, dim>, n_embeddings>;
  using Side = std::array<Slot, 6>;

  std::array<Side, 2> sides;

  static void fill_slot(const auto &net, auto &slot,
                        const PKMN::Pokemon &pokemon) {
    std::array<float, Encode::Pokemon::n_dim> input;

    for (auto s = 0; s < n_status; ++s) {
      for (uint8_t b = 0; b < 16; ++b) {
        auto copy = pokemon;
        // set moves
        for (auto m = 0; m < 4; ++m) {
          copy.moves[m].pp = b & (1 << m);
        }
        Encode::Pokemon::write(copy, 0, input.data());
        // set status index
        if (s > 0) {
          input[Encode::Stats::n_dim + Encode::MoveSlots::n_dim + (s - 1)] = 1;
        }
        auto key = 16 * s + b;
        net.propagate(input.data(), slot[key].data());
      }
    }
  }

  void fill(const auto &net, const PKMN::Battle &battle) {
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        fill_slot(net, sides[s][p], battle.sides[s].pokemon[p]);
      }
    }
  }
};

} // namespace NN
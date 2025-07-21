#pragma once

#include <encode/battle.h>
#include <libpkmn/data.h>
#include <nn/subnet.h>

namespace NN {

struct Network {

  static constexpr auto pokemon_hidden_dim = 32;
  static constexpr auto pokemon_out_dim = 39;
  static constexpr auto active_hidden_dim = 32;
  static constexpr auto active_out_dim = 55;
  static constexpr auto side_out_dim = 256;

  // leading dims is hp percent
  static_assert((1 + active_out_dim) + 5 * (1 + pokemon_out_dim) ==
                side_out_dim);

  using PokemonSubnet =
      EmbeddingNet<Encode::Pokemon::n_dim, pokemon_hidden_dim, pokemon_out_dim>;
  using ActiveSubnet =
      EmbeddingNet<Encode::Active::n_dim, active_hidden_dim, active_out_dim>;

  prng device{};
  PokemonSubnet p;
  ActiveSubnet a;
  MainNet m;

  bool read_parameters(std::istream &stream) {
    return p.read_parameters(stream) && a.read_parameters(stream) &&
           m.read_parameters(stream);
  }

  float inference(const pkmn_gen1_battle &battle,
                  const pkmn_gen1_chance_durations &durations) {
    static thread_local float pokemon_input[2][5][Encode::Pokemon::n_dim];
    static thread_local float active_input[2][1][Encode::Active::n_dim];
    static thread_local float pokemon_output[2][5][pokemon_out_dim];
    static thread_local float active_output[2][1][active_out_dim];
    static thread_local float main_input[256][2];

    const auto &b = View::ref(battle);

    for (auto s = 0; s < 2; ++s) {
      const auto &dur = View::ref(durations).get(s);
      const auto &side = b.sides[s];

      auto k = 0;
      const auto &active = side.stored();
      for (k = 1; k < 6; ++k) {
      }
    }
    return 0;
  }
};

} // namespace NN
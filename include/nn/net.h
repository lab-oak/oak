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
  static constexpr auto main_hidden_dim = 64;
  static constexpr auto policy_dim = 151 + 164; // no Struggle, None
  // leading dims "1 + " are hp percentage.
  static_assert((1 + active_out_dim) + 5 * (1 + pokemon_out_dim) ==
                side_out_dim);

  using PokemonSubnet =
      EmbeddingNet<Encode::Pokemon::n_dim, pokemon_hidden_dim, pokemon_out_dim>;
  using ActiveSubnet =
      EmbeddingNet<Encode::Active::n_dim, active_hidden_dim, active_out_dim>;

  mt19937 device;
  PokemonSubnet p;
  ActiveSubnet a;
  MainNet<main_hidden_dim, policy_dim> m;

  Network() : device{std::random_device{}()}, p{}, a{}, m{} {
    p.initialize(device);
    a.initialize(device);
    m.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return p.read_parameters(stream) && a.read_parameters(stream) &&
           m.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return p.write_parameters(stream) && a.write_parameters(stream) &&
           m.write_parameters(stream);
  }

  static constexpr auto main_input_index(auto i) noexcept {
    assert(i > 0);
    return 1 + active_out_dim + (i - 1) * (1 + pokemon_out_dim);
  }

  static auto print_main_input(float *input) {
    for (auto s = 0; s < 2; ++s) {
      std::cout << "Active " << (int)(100 * input[0]) << "%\n";
      for (auto i = 0; i < active_out_dim; ++i) {
        std::cout << input[1 + i] << ' ';
      }
      std::cout << std::endl;

      for (auto p = 1; p < 6; ++p) {
        auto p_input = input + main_input_index(p);
        std::cout << "Pokemon " << (int)(100 * p_input[0]) << "%\n";
        for (auto i = 0; i < pokemon_out_dim; ++i) {
          std::cout << p_input[1 + i] << ' ';
        }
        std::cout << std::endl;
      }

      input += 256;
    }
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    static thread_local float pokemon_input[2][5][Encode::Pokemon::n_dim];
    static thread_local float active_input[2][1][Encode::Active::n_dim];
    static thread_local float main_input[2][256];

    const auto &battle = View::ref(b);
    const auto &durations = View::ref(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (!stored.hp) {
        std::memset(main_input[s], 0, (1 + active_out_dim) * sizeof(float));
      } else {
        Encode::Active::write(stored, side.active, duration,
                              active_input[s][0]);
        a.propagate(active_input[s][0], main_input[s] + 1);
        main_input[s][0] = (float)stored.hp / stored.stats.hp;
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        float *input = pokemon_input[s][slot - 1];
        float *output = main_input[s] + main_input_index(slot - 1);
        const auto &pokemon = side.get(slot);
        if (!pokemon.hp) {
          std::memset(output, 0, (1 + pokemon_out_dim) * sizeof(float));
        } else {
          const auto sleep = duration.sleep(slot - 1);
          Encode::Pokemon::write(pokemon, sleep, input);
          p.propagate(input, output + 1);
          output[0] = (float)pokemon.hp / pokemon.stats.hp;
        }
      }
    }

    // print_main_input(main_input[0]);
    // std::cout << std::endl;

    float *policy_output = nullptr;
    float value = m.propagate(main_input[0], policy_output);

    return value;
  }

  bool operator==(const Network &other) {
    return (p == p) && (a == a) && (m == m);
  }
};

} // namespace NN
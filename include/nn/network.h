#pragma once

#include <encode/battle.h>
#include <libpkmn/data.h>
#include <nn/cache.h>
#include <nn/params.h>
#include <nn/subnet.h>

namespace NN {

struct Network {
  using PokemonNet =
      EmbeddingNet<Encode::Pokemon::n_dim, pokemon_hidden_dim, pokemon_out_dim>;
  using ActiveNet =
      EmbeddingNet<Encode::Active::n_dim, active_hidden_dim, active_out_dim>;

  mt19937 device;
  PokemonNet pokemon_net;
  std::array<std::array<PokemonCache<float, pokemon_out_dim>, 6>, 2>
      pokemon_cache;
  ActiveNet active_net;
  MainNet<2 * side_out_dim, hidden_dim, value_hidden_dim, policy_hidden_dim,
          policy_out_dim>
      main_net;

  Network()
      : device{std::random_device{}()}, pokemon_net{}, active_net{},
        main_net{} {}

  bool operator==(const Network &other) {
    return (pokemon_net == pokemon_net) && (active_net == active_net) &&
           (main_net == main_net);
  }

  void initialize() {
    pokemon_net.initialize(device);
    active_net.initialize(device);
    main_net.initialize(device);
  }

  void fill_cache(const pkmn_gen1_battle &b) {
    const auto &battle = View::ref(b);
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        pokemon_cache[s][p].fill(pokemon_net, battle.sides[s].pokemon[p]);
      }
    }
  }

  bool read_parameters(std::istream &stream) {
    return pokemon_net.read_parameters(stream) &&
           active_net.read_parameters(stream) &&
           main_net.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return pokemon_net.write_parameters(stream) &&
           active_net.write_parameters(stream) &&
           main_net.write_parameters(stream);
  }

  static constexpr auto main_input_index(auto i) noexcept {
    assert(i > 0);
    return 1 + active_out_dim + (i - 1) * (1 + pokemon_out_dim);
  }

  // bool defer(const pkmn_gen1_battle &b) const {
  //   auto n_alive = 0;
  //   const auto &battle = View::ref(b);
  //   for (auto s = 0; s < 2; ++s) {
  //     const auto &side = battle.sides[s];
  //     for (const auto &pokemon : side.pokemon) {
  //       n_alive += (pokemon.hp > 0);
  //     }
  //   }
  //   return (n_alive <= 4);
  // }

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

      Encode::Active::write(stored, side.active, duration, active_input[s][0]);
      active_net.propagate(active_input[s][0], main_input[s] + 1);
      main_input[s][0] = (float)stored.hp / stored.stats.hp;

      for (auto slot = 2; slot <= 6; ++slot) {
        // float *input = pokemon_input[s][slot - 2];
        float *output = main_input[s] + main_input_index(slot - 1);
        const auto &pokemon = side.get(slot);

        const auto sleep = duration.sleep(slot - 1);
        // Encode::Pokemon::write(pokemon, sleep, input);
        const auto *embedding =
            pokemon_cache[s][side.order[slot - 1] - 1].get(pokemon, sleep);
        std::memcpy(output + 1, embedding, pokemon_out_dim * sizeof(float));
        // pokemon_net.propagate(input, output + 1);
        output[0] = (float)pokemon.hp / pokemon.stats.hp;
      }
    }

    float value = main_net.propagate(main_input[0]);
    return value;
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d, const auto m,
                  const auto n, const auto *p1_choice_index,
                  const auto *p2_choice_index, float *p1, float *p2) {
    static thread_local float pokemon_input[2][5][Encode::Pokemon::n_dim];
    static thread_local float active_input[2][1][Encode::Active::n_dim];
    static thread_local float main_input[2][256];

    const auto &battle = View::ref(b);
    const auto &durations = View::ref(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      Encode::Active::write(stored, side.active, duration, active_input[s][0]);
      active_net.propagate(active_input[s][0], main_input[s] + 1);
      main_input[s][0] = (float)stored.hp / stored.stats.hp;

      for (auto slot = 2; slot <= 6; ++slot) {
        // float *input = pokemon_input[s][slot - 2];
        float *output = main_input[s] + main_input_index(slot - 1);
        const auto &pokemon = side.get(slot);

        const auto sleep = duration.sleep(slot - 1);
        // Encode::Pokemon::write(pokemon, sleep, input);
        const auto *embedding =
            pokemon_cache[s][side.order[slot - 1] - 1].get(pokemon, sleep);
        std::memcpy(output + 1, embedding, pokemon_out_dim * sizeof(float));
        // pokemon_net.propagate(input, output + 1);
        output[0] = (float)pokemon.hp / pokemon.stats.hp;
      }
    }

    float value = main_net.propagate(main_input[0], m, n, p1_choice_index,
                                     p2_choice_index, p1, p2);
    return value;
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
};

} // namespace NN
#pragma once

#include <encode/battle/battle.h>
#include <libpkmn/data.h>
#include <nn/battle/cache.h>
#include <nn/params.h>
#include <nn/subnet.h>
#include <util/random.h>

namespace NN {

namespace Battle {

template <int in_dim, int hidden_dim, int value_hidden_dim,
          int policy_hidden_dim, int policy_out_dim>
struct MainNet {

  Affine<in_dim, hidden_dim> fc0;
  Affine<hidden_dim, value_hidden_dim> value_fc1;
  Affine<value_hidden_dim, 1, false> value_fc2;
  Affine<hidden_dim, policy_hidden_dim> policy1_fc1;
  Affine<policy_hidden_dim, policy_out_dim, false> policy1_fc2;
  Affine<hidden_dim, policy_hidden_dim> policy2_fc1;
  Affine<policy_hidden_dim, policy_out_dim, false> policy2_fc2;

  bool operator==(const MainNet &) const = default;

  void initialize(auto &device) {
    fc0.initialize(device);
    value_fc1.initialize(device);
    value_fc2.initialize(device);
    policy1_fc1.initialize(device);
    policy1_fc2.initialize(device);
    policy2_fc1.initialize(device);
    policy2_fc2.initialize(device);
  }

  bool read_parameters(std::istream &stream) {
    return fc0.read_parameters(stream) && value_fc1.read_parameters(stream) &&
           value_fc2.read_parameters(stream) &&
           policy1_fc1.read_parameters(stream) &&
           policy1_fc2.read_parameters(stream) &&
           policy2_fc1.read_parameters(stream) &&
           policy2_fc2.read_parameters(stream);
  }

  bool write_parameters(std::ostream &stream) {
    return fc0.write_parameters(stream) && value_fc1.write_parameters(stream) &&
           value_fc2.write_parameters(stream) &&
           policy1_fc1.write_parameters(stream) &&
           policy1_fc2.write_parameters(stream) &&
           policy2_fc1.write_parameters(stream) &&
           policy2_fc2.write_parameters(stream);
  }

  // TODO for now main does not do policy out, thats done by full net
  float propagate(const float *input_data) const {
    static thread_local float buffer0[hidden_dim];
    static thread_local float value_buffer1[value_hidden_dim];
    static thread_local float value_buffer2[1];
    fc0.propagate(input_data, buffer0);
    value_fc1.propagate(buffer0, value_buffer1);
    value_fc2.propagate(value_buffer1, value_buffer2);
    return 1 / (1 + std::exp(-value_buffer2[0]));
  }

  float propagate(const float *input_data, const auto m, const auto n,
                  const auto *p1_choice_index, const auto *p2_choice_index,
                  float *p1, float *p2) const {
    static thread_local float buffer0[hidden_dim];
    static thread_local float value_buffer1[value_hidden_dim];
    static thread_local float value_buffer2[1];
    static thread_local float policy1_buffer1[policy_hidden_dim];
    static thread_local float policy2_buffer1[policy_hidden_dim];

    fc0.propagate(input_data, buffer0);
    value_fc1.propagate(buffer0, value_buffer1);
    value_fc2.propagate(value_buffer1, value_buffer2);
    policy1_fc1.propagate(buffer0, policy1_buffer1);
    policy2_fc1.propagate(buffer0, policy2_buffer1);

    for (auto i = 0; i < m; ++i) {
      const auto p1_c = p1_choice_index[i];
      assert(p1_c < Encode::Policy::n_dim);
      const float logit =
          policy1_fc2.weights.row(p1_c).dot(Eigen::Map<const Eigen::VectorXf>(
              policy1_buffer1, value_hidden_dim)) +
          policy1_fc2.biases[p1_c];
      p1[i] = logit;
    }

    for (auto i = 0; i < n; ++i) {
      const auto p2_c = p2_choice_index[i];
      assert(p2_c < Encode::Policy::n_dim);
      const float logit =
          policy2_fc2.weights.row(p2_c).dot(Eigen::Map<const Eigen::VectorXf>(
              policy2_buffer1, value_hidden_dim)) +
          policy2_fc2.biases[p2_c];
      p2[i] = logit;
    }

    return 1 / (1 + std::exp(-value_buffer2[0]));
  }
};

struct Network {
  using PokemonNet =
      EmbeddingNet<Encode::Pokemon::n_dim, pokemon_hidden_dim, pokemon_out_dim>;
  using ActiveNet =
      EmbeddingNet<Encode::Active::n_dim, active_hidden_dim, active_out_dim>;

  PokemonNet pokemon_net;
  std::array<std::array<PokemonCache<float, pokemon_out_dim>, 6>, 2>
      pokemon_cache;
  ActiveNet active_net;
  MainNet<2 * side_out_dim, hidden_dim, value_hidden_dim, policy_hidden_dim,
          policy_out_dim>
      main_net;
  mt19937 device;

  Network() : pokemon_net{}, active_net{}, main_net{} {}

  bool operator==(const Network &other) {
    return (pokemon_net == pokemon_net) && (active_net == active_net) &&
           (main_net == main_net);
  }

  void initialize() {
    mt19937 device{std::random_device{}()};
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

  void write_main(float main_input[2][256], const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    static thread_local float pokemon_input[2][5][Encode::Pokemon::n_dim];
    static thread_local float active_input[2][1][Encode::Active::n_dim];

    const auto &battle = View::ref(b);
    const auto &durations = View::ref(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (stored.hp == 0) {
        std::fill(main_input[s], main_input[s] + (Encode::Active::n_dim + 1),
                  0);
      } else {
        Encode::Active::write(stored, side.active, duration,
                              active_input[s][0]);
        active_net.propagate(active_input[s][0], main_input[s] + 1);
        main_input[s][0] = (float)stored.hp / stored.stats.hp;
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        float *output = main_input[s] + main_input_index(slot - 1);
        const auto id = side.order[slot - 1];
        if (id == 0) {
          std::fill(output, output + (pokemon_out_dim + 1), 0);
        } else {
          const auto &pokemon = side.pokemon[id - 1];
          if (pokemon.hp == 0) {
            std::fill(output, output + (pokemon_out_dim + 1), 0);
          } else {
            const auto sleep = duration.sleep(slot - 1);
            const auto *embedding =
                pokemon_cache[s][side.order[slot - 1] - 1].get(pokemon, sleep);
            std::memcpy(output + 1, embedding, pokemon_out_dim * sizeof(float));
            output[0] = (float)pokemon.hp / pokemon.stats.hp;
          }
        }
      }
    }
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    static thread_local float main_input[2][256];
    write_main(main_input, b, d);
    float value = main_net.propagate(main_input[0]);
    return value;
  }

  void foo(const pkmn_gen1_battle &b, const auto &p1_choices,
           const auto &p2_choices) {
    // const auto &battle = View::ref(b);

    // std::array<uint16_t, 9> p1_choice_indices;
    // std::array<uint16_t, 9> p2_choice_indices;

    // for (auto i = 0; i < m; ++i) {
    //   p1_choice_indices[i] =
    //       Encode::Policy::get_index(b.sides[0], p1_choices[i]);
    // }

    // for (auto i = 0; i < n; ++i) {
    //   p2_choice_indices[i] =
    //       Encode::Policy::get_index(b.sides[1], p2_choices[i]);
    // }
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d, const auto m,
                  const auto n, const auto *p1_choice_index,
                  const auto *p2_choice_index, float *p1, float *p2) {
    static thread_local float main_input[2][256];
    write_main(main_input, b, d);
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

} // namespace Battle

} // namespace NN
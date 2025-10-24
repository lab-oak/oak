#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <nn/battle/cache.h>
#include <nn/battle/stockfish/network.h>
#include <nn/params.h>
#include <nn/subnet.h>
#include <util/random.h>

namespace NN {

namespace Battle {

struct MainNet {

  Affine<> fc0;
  Affine<> value_fc1;
  Affine<false> value_fc2;
  Affine<> p1_policy_fc1;
  Affine<false> p1_policy_fc2;
  Affine<> p2_policy_fc1;
  Affine<false> p2_policy_fc2;

  std::vector<float> buffer;
  std::vector<float> value_buffer;
  std::vector<float> p1_policy_buffer;
  std::vector<float> p2_policy_buffer;

  MainNet(uint32_t in_dim, uint32_t hidden_dim, uint32_t value_hidden_dim,
          uint32_t policy_hidden_dim, uint32_t policy_out_dim)
      : fc0(in_dim, hidden_dim), value_fc1(hidden_dim, value_hidden_dim),
        value_fc2(value_hidden_dim, 1),
        p1_policy_fc1(hidden_dim, policy_hidden_dim),
        p1_policy_fc2(policy_hidden_dim, policy_out_dim),
        p2_policy_fc1(hidden_dim, policy_hidden_dim),
        p2_policy_fc2(policy_hidden_dim, policy_out_dim), buffer{},
        value_buffer{}, p1_policy_buffer{}, p2_policy_buffer{} {
    buffer.resize(hidden_dim);
    value_buffer.resize(value_hidden_dim);
    p1_policy_buffer.resize(policy_hidden_dim);
    p2_policy_buffer.resize(policy_hidden_dim);
  }

  bool operator==(const MainNet &) const = default;

  bool read_parameters(std::istream &stream) {
    const bool ok = fc0.read_parameters(stream) &&
                    value_fc1.read_parameters(stream) &&
                    value_fc2.read_parameters(stream) &&
                    p1_policy_fc1.read_parameters(stream) &&
                    p1_policy_fc2.read_parameters(stream) &&
                    p2_policy_fc1.read_parameters(stream) &&
                    p2_policy_fc2.read_parameters(stream);

    if (!ok) {
      return false;
    }
    buffer.resize(fc0.out_dim);
    value_buffer.resize(value_fc1.out_dim);
    p1_policy_buffer.resize(p1_policy_fc1.out_dim);
    p2_policy_buffer.resize(p2_policy_fc1.out_dim);
    return true;
  }

  bool write_parameters(std::ostream &stream) const {
    return fc0.write_parameters(stream) && value_fc1.write_parameters(stream) &&
           value_fc2.write_parameters(stream) &&
           p1_policy_fc1.write_parameters(stream) &&
           p1_policy_fc2.write_parameters(stream) &&
           p2_policy_fc1.write_parameters(stream) &&
           p2_policy_fc2.write_parameters(stream);
  }

  float propagate(const float *input_data) {
    float output;
    fc0.propagate(input_data, buffer.data());
    value_fc1.propagate(buffer.data(), value_buffer.data());
    value_fc2.propagate(value_buffer.data(), &output);
    return 1 / (1 + std::exp(-output));
  }

  float propagate(const float *input_data, const auto m, const auto n,
                  const auto *p1_choice_index, const auto *p2_choice_index,
                  float *p1, float *p2) {
    float output;
    fc0.propagate(input_data, buffer.data());
    value_fc1.propagate(buffer.data(), value_buffer.data());
    value_fc2.propagate(value_buffer.data(), &output);

    p1_policy_fc1.propagate(buffer.data(), p1_policy_buffer.data());
    p2_policy_fc1.propagate(buffer.data(), p2_policy_buffer.data());

    std::vector<float> p1_policy_buffer_2{};
    std::vector<float> p2_policy_buffer_2{};
    p1_policy_buffer_2.resize(Encode::Battle::Policy::n_dim);
    p2_policy_buffer_2.resize(Encode::Battle::Policy::n_dim);

    p1_policy_fc2.propagate(p1_policy_buffer.data(), p1_policy_buffer_2.data());
    p2_policy_fc2.propagate(p2_policy_buffer.data(), p2_policy_buffer_2.data());

    for (auto i = 0; i < m; ++i) {
      const auto p1_c = p1_choice_index[i];
      assert(p1_c < Encode::Battle::Policy::n_dim);
      // const float logit =
      //     p1_policy_fc2.weights.row(p1_c).dot(Eigen::Map<const
      //     Eigen::VectorXf>(
      //         p1_policy_buffer.data(), p1_policy_fc1.out_dim)) +
      //     p1_policy_fc2.biases[p1_c];
      // p1[i] = logit;
      p1[i] = p1_policy_buffer_2[p1_c];
    }

    for (auto i = 0; i < n; ++i) {
      const auto p2_c = p2_choice_index[i];
      assert(p2_c < Encode::Battle::Policy::n_dim);
      // const float logit =
      //     p2_policy_fc2.weights.row(p2_c).dot(Eigen::Map<const
      //     Eigen::VectorXf>(
      //         p2_policy_buffer.data(), p2_policy_fc1.out_dim)) +
      //     p2_policy_fc2.biases[p2_c];
      // p2[i] = logit;
      p2[i] = p2_policy_buffer_2[p2_c];
    }

    return 1 / (1 + std::exp(-output));
  }
};

struct Network {
  using PokemonNet = EmbeddingNet<>;
  using ActiveNet = EmbeddingNet<>;

  PokemonNet pokemon_net;
  std::array<std::array<PokemonCache<float, pokemon_out_dim>, 6>, 2>
      pokemon_cache;
  ActiveNet active_net;

  MainNet main_net;
  mt19937 device;

  std::array<std::array<PokemonCache<uint8_t, pokemon_out_dim>, 6>, 2>
      discrete_pokemon_cache;
  std::array<std::array<ActivePokemonCache<uint8_t>, 6>, 2>
      discrete_active_cache;
  Stockfish::NetworkArchitecture discrete_main_net;
  bool use_discrete = false;

  Network(uint32_t phd = pokemon_hidden_dim, uint32_t ahd = active_hidden_dim,
          uint32_t hd = hidden_dim, uint32_t vhd = value_hidden_dim,
          uint32_t pohd = policy_hidden_dim)
      : pokemon_net{Encode::Battle::Pokemon::n_dim, phd, pokemon_out_dim},
        active_net{Encode::Battle::Active::n_dim, ahd, active_out_dim},
        main_net{2 * side_out_dim, hd, vhd, pohd, policy_out_dim},
        discrete_active_cache{} {}

  bool operator==(const Network &other) {
    return (pokemon_net == pokemon_net) && (active_net == active_net) &&
           (main_net == main_net);
  }

  void fill_cache(const pkmn_gen1_battle &b) {
    const auto &battle = PKMN::view(b);
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        pokemon_cache[s][p].fill(pokemon_net, battle.sides[s].pokemon[p]);

        for (auto j = 0; j < PokemonCache<float>::n_embeddings; ++j) {
          for (auto i = 0; i < pokemon_out_dim; ++i) {
            discrete_pokemon_cache[s][p].embeddings[j][i] =
                pokemon_cache[s][p].embeddings[j][i] * 127;
            // std::cout << pokemon_cache[s][p].embeddings[j][i] << '/';
            // std::cout << discrete_pokemon_cache[s][p].embeddings[j][i] /
            // 127.0 << ' ';
          }
          // std::cout << std::endl;
        }
      }
    }
  }

  bool read_parameters(std::istream &stream) {
    const bool ok = pokemon_net.read_parameters(stream) &&
                    active_net.read_parameters(stream) &&
                    main_net.read_parameters(stream);
    if (!ok) {
      return false;
    }
    char dummy;
    if (stream.read(&dummy, 1)) {
      return false;
    } else {
      discrete_main_net.copy_parameters(main_net);
      return true;
    }
  }

  bool write_parameters(std::ostream &stream) const {
    return pokemon_net.write_parameters(stream) &&
           active_net.write_parameters(stream) &&
           main_net.write_parameters(stream);
  }

  static constexpr auto main_input_index(auto i) noexcept {
    assert(i > 0);
    return 1 + active_out_dim + (i - 1) * (1 + pokemon_out_dim);
  }

  void write_main(float main_input[2][side_out_dim], const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    static thread_local float active_input[2][1][Encode::Battle::Active::n_dim];

    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (stored.hp == 0) {
        std::fill(main_input[s], main_input[s] + (active_out_dim + 1), 0);
      } else {
        std::fill(active_input[s][0],
                  active_input[s][0] + Encode::Battle::Active::n_dim, 0);
        Encode::Battle::Active::write(stored, side.active, duration,
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

  void write_main(uint8_t main_input[2][side_out_dim],
                  const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      if (stored.hp == 0) {
        std::fill(main_input[s], main_input[s] + (active_out_dim + 1), 0);
      } else {
        const auto *embedding = discrete_active_cache[s][side.order[0] - 1].get(
            active_net, side.active, stored, duration);
        std::memcpy(main_input[s] + 1, embedding,
                    active_out_dim * sizeof(uint8_t));

        main_input[s][0] = (float)stored.hp / stored.stats.hp;
      }

      for (auto slot = 2; slot <= 6; ++slot) {
        uint8_t *output = main_input[s] + main_input_index(slot - 1);
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
                discrete_pokemon_cache[s][side.order[slot - 1] - 1].get(pokemon,
                                                                        sleep);
            std::memcpy(output + 1, embedding, pokemon_out_dim * sizeof(float));
            output[0] = (float)pokemon.hp / pokemon.stats.hp;
          }
        }
      }
    }
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    static thread_local float main_input[2][side_out_dim];
    static thread_local uint8_t discrete_main_input[2][side_out_dim];

    write_main(main_input, b, d);
    write_main(discrete_main_input, b, d);

    std::cout << "f32\n";
    print_main_input(main_input);
    std::cout << "u8\n";
    print_main_input(discrete_main_input);

    float value = main_net.propagate(main_input[0]);
    return value;
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d, const auto m,
                  const auto n, const auto *p1_choice, const auto *p2_choice,
                  float *p1, float *p2) {
    static thread_local float main_input[2][side_out_dim];
    static thread_local uint16_t p1_choice_index[9];
    static thread_local uint16_t p2_choice_index[9];
    write_main(main_input, b, d);
    const auto &battle = PKMN::view(b);
    for (auto i = 0; i < m; ++i) {
      p1_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[0], p1_choice[i]);
    }
    for (auto i = 0; i < n; ++i) {
      p2_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[1], p2_choice[i]);
    }
    float value = main_net.propagate(main_input[0], m, n, p1_choice_index,
                                     p2_choice_index, p1, p2);
    return value;
  }

  static void print_main_input(float input[2][256]) {
    for (auto s = 0; s < 2; ++s) {
      std::cout << "Active " << (int)(100 * input[s][0]) << "%\n";
      for (auto i = 0; i < active_out_dim; ++i) {
        std::cout << input[s][1 + i] << ' ';
      }
      std::cout << std::endl;

      for (auto p = 1; p < 6; ++p) {
        auto p_input = input[s] + main_input_index(p);
        std::cout << "Pokemon " << (int)(100 * p_input[0]) << "%\n";
        for (auto i = 0; i < pokemon_out_dim; ++i) {
          std::cout << p_input[1 + i] << ' ';
        }
        std::cout << std::endl;
      }
    }
  }

  static void print_main_input(uint8_t input[2][256]) {
    for (auto s = 0; s < 2; ++s) {
      std::cout << "Active " << (int)(100 * input[s][0]) << "%\n";
      for (auto i = 0; i < active_out_dim; ++i) {
        std::cout << input[s][1 + i] / 127.0 << ' ';
      }
      std::cout << std::endl;

      for (auto p = 1; p < 6; ++p) {
        auto p_input = input[s] + main_input_index(p);
        std::cout << "Pokemon " << (int)(100 * p_input[0]) << "%\n";
        for (auto i = 0; i < pokemon_out_dim; ++i) {
          std::cout << p_input[1 + i] / 127.0 << ' ';
        }
        std::cout << std::endl;
      }
    }
  }
};

} // namespace Battle

} // namespace NN
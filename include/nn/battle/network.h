#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <nn/battle/cache.h>
#include <nn/battle/main-net.h>
#include <nn/battle/stockfish/network.h>
#include <nn/default-hyperparameters.h>
#include <nn/embedding-net.h>
#include <util/random.h>

namespace NN::Battle {

inline constexpr float sigmoid(const float x) { return 1 / (1 + std::exp(-x)); }

std::array<uint8_t, 5> &random_permutation_5() {
  static thread_local std::array<uint8_t, 5> indices{1, 2, 3, 4, 5};
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::shuffle(indices.begin(), indices.end(), rng);
  return indices;
}

struct Network {

  bool use_discrete;

  EmbeddingNet<> pokemon_net;
  EmbeddingNet<> active_net;
  MainNet main_net;
  std::shared_ptr<Stockfish::Network> discrete_main_net;

  BattleCaches<float> battle_cache;
  BattleCaches<uint8_t> discrete_battle_cache;

  uint32_t pokemon_out_dim;
  uint32_t active_out_dim;
  uint32_t side_embedding_dim;
  std::vector<float> battle_embedding;
  std::vector<uint8_t> battle_embedding_d;

  void fill_pokemon_caches(const pkmn_gen1_battle &b) {
    const auto &battle = PKMN::view(b);
    for (auto s = 0; s < 2; ++s) {
      for (auto p = 0; p < 6; ++p) {
        battle_cache.pokemon[s][p].fill(pokemon_net,
                                        battle.sides[s].pokemon[p]);
        // rather than calling fill again, just copy
        for (auto j = 0; j < PokemonCache<float>::n_embeddings; ++j) {
          for (auto i = 0; i < pokemon_out_dim; ++i) {
            discrete_battle_cache.pokemon[s][p].embeddings[j][i] =
                battle_cache.pokemon[s][p].embeddings[j][i] * 127;
          }
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
      pokemon_out_dim = pokemon_net.fc1.out_dim;
      active_out_dim = active_net.fc1.out_dim;
      side_embedding_dim = (1 + active_out_dim) + 5 * (1 + pokemon_out_dim);
      battle_embedding.resize(2 * side_embedding_dim);
      battle_embedding_d.resize(2 * side_embedding_dim);
      battle_cache = BattleCaches<float>{pokemon_out_dim, active_out_dim};
      discrete_battle_cache =
          BattleCaches<uint8_t>{pokemon_out_dim, active_out_dim};
      return true;
    }
  }

  bool write_parameters(std::ostream &stream) const {
    return pokemon_net.write_parameters(stream) &&
           active_net.write_parameters(stream) &&
           main_net.write_parameters(stream);
  }

  auto side_embedding_index(auto i) const noexcept {
    assert(i > 0);
    return (1 + active_out_dim) + (i - 1) * (1 + pokemon_out_dim);
  }

  void enable_discrete() {
    discrete_main_net = Stockfish::make_network(
        main_net.fc0.in_dim, main_net.fc0.out_dim, main_net.value_fc2.out_dim);
    discrete_main_net->copy_parameters(main_net.fc0, main_net.fc1,
                                       main_net.value_fc2, main_net.value_fc3);
    use_discrete = true;
  }

  template <typename T>
  void write_battle_embedding(T *battle_embedding, const pkmn_gen1_battle &b,
                              const pkmn_gen1_chance_durations &d) {
    const auto &battle = PKMN::view(b);
    const auto &durations = PKMN::view(d);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = battle.sides[s];
      const auto &duration = durations.get(s);
      const auto &stored = side.stored();

      auto *side_embedding = battle_embedding + s * side_embedding_index(6);

      if (stored.hp == 0) {
        std::fill(side_embedding, side_embedding + (active_out_dim + 1), 0);
      } else {
        if constexpr (std::is_integral_v<T>) {
          const T *embedding =
              discrete_battle_cache.active[s][side.order[0] - 1].get(
                  active_net, side.active, stored, duration);
          std::copy(embedding, embedding + active_out_dim, side_embedding + 1);
          side_embedding[0] = (float)stored.hp / stored.stats.hp * 127;
        } else {
          const T *embedding = battle_cache.active[s][side.order[0] - 1].get(
              active_net, side.active, stored, duration);
          std::copy(embedding, embedding + active_out_dim, side_embedding + 1);
          side_embedding[0] = (float)stored.hp / stored.stats.hp;
        }
      }

      const auto &permute = random_permutation_5();

      for (auto slot = 2; slot <= 6; ++slot) {
        auto *slot_embedding = side_embedding + side_embedding_index(slot - 1);
        // const auto id = side.order[permute[slot - 2]];
        const auto id = side.order[slot - 1];

        if (id == 0) {
          std::fill(slot_embedding, slot_embedding + (pokemon_out_dim + 1), 0);
        } else {
          const auto &pokemon = side.pokemon[id - 1];
          if (pokemon.hp == 0) {
            std::fill(slot_embedding, slot_embedding + (pokemon_out_dim + 1),
                      0);
          } else {
            const auto sleep = duration.sleep(slot - 1);
            if constexpr (std::is_integral_v<T>) {
              const T *embedding =
                  discrete_battle_cache.pokemon[s][id - 1].get(pokemon, sleep);
              std::copy(embedding, embedding + pokemon_out_dim,
                        slot_embedding + 1);
              slot_embedding[0] = (float)pokemon.hp / pokemon.stats.hp * 127;
            } else {
              const T *embedding =
                  battle_cache.pokemon[s][id - 1].get(pokemon, sleep);
              std::copy(embedding, embedding + pokemon_out_dim,
                        slot_embedding + 1);
              slot_embedding[0] = (float)pokemon.hp / pokemon.stats.hp;
            }
          }
        }
      }
    }
  }

  float inference(const pkmn_gen1_battle &b,
                  const pkmn_gen1_chance_durations &d) {
    float value;
    if (use_discrete) {
      write_battle_embedding<uint8_t>(battle_embedding_d.data(), b, d);
      value = sigmoid(discrete_main_net->propagate(battle_embedding_d.data()));
    } else {
      write_battle_embedding<float>(battle_embedding.data(), b, d);
      value = sigmoid(main_net.propagate(battle_embedding.data()));
    }
    assert(!std::isnan(value));
    return value;
  }

  template <bool use_value = true>
  auto inference(const pkmn_gen1_battle &b, const pkmn_gen1_chance_durations &d,
                 const auto m, const auto n, const auto *p1_choice,
                 const auto *p2_choice, float *p1, float *p2)
      -> std::conditional_t<use_value, float, void> {

    static thread_local uint16_t p1_choice_index[9];
    static thread_local uint16_t p2_choice_index[9];

    const auto &battle = PKMN::view(b);
    for (auto i = 0; i < m; ++i) {
      p1_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[0], p1_choice[i]);
    }
    for (auto i = 0; i < n; ++i) {
      p2_choice_index[i] =
          Encode::Battle::Policy::get_index(battle.sides[1], p2_choice[i]);
    }

    float value;
    if constexpr (use_value) {
      if (use_discrete) {
        throw std::runtime_error(
            "Value + Policy output with discrete nets not supported.");
        return -1;
      }
      write_battle_embedding<float>(battle_embedding.data(), b, d);
      value =
          main_net.propagate<true>(battle_embedding.data(), m, n,
                                   p1_choice_index, p2_choice_index, p1, p2);
      return sigmoid(value);
    } else {
      write_battle_embedding<float>(battle_embedding.data(), b, d);
      // won't use value layers or return anything
      main_net.propagate<false>(battle_embedding.data(), m, n, p1_choice_index,
                                p2_choice_index, p1, p2);
    }
  }

  void print_battle_embedding(const float *input) const {
    for (auto s = 0; s < 2; ++s) {
      const auto *side_embedding = input + s * side_embedding_index(6);
      std::cout << "Active " << (int)(100 * side_embedding[0]) << "%\n";
      for (auto i = 0; i < active_out_dim; ++i) {
        std::cout << side_embedding[1 + i] << ' ';
      }
      std::cout << std::endl;

      for (auto p = 1; p < 6; ++p) {
        const auto *p_input = side_embedding + side_embedding_index(p);
        std::cout << "Pokemon " << (int)(100 * p_input[0]) << "%\n";
        for (auto i = 0; i < pokemon_out_dim; ++i) {
          std::cout << p_input[1 + i] << ' ';
        }
        std::cout << std::endl;
      }
    }
  }
};

} // namespace NN::Battle
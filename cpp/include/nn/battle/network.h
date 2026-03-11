#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/policy.h>
#include <nn/battle/cache.h>
#include <nn/battle/main-net.h>
#include <nn/battle/quantized/network.h>
#include <nn/default-hyperparameters.h>
#include <nn/embedding-net.h>
#include <util/random.h>

namespace NN::Battle {

inline constexpr float sigmoid(const float x) { return 1 / (1 + std::exp(-x)); }

struct NetworkBase {
  virtual void init_caches(const pkmn_gen1_battle &b) = 0;
  virtual bool read_parameters(const std::istream &stream) = 0;
  virtual void write_parameters(std::ostream &stream) = 0;
  virtual bool check_cache() const = 0;
  virtual float value_inference(const pkmn_gen1_battle &b,
                                const pkmn_gen1_chance_durations &d) = 0;
  virtual void policy_inference(const pkmn_gen1_battle &b,
                                const pkmn_gen1_chance_durations &d,
                                const auto m, const auto n,
                                const auto *p1_choice, const auto *p2_choice,
                                float *p1, float *p2) = 0;
  virtual float value_policy_inference(const pkmn_gen1_battle &b,
                                       const pkmn_gen1_chance_durations &d,
                                       const auto m, const auto n,
                                       const auto *p1_choice,
                                       const auto *p2_choice, float *p1,
                                       float *p2) = 0;
};

template <typename MainNet> class Network : public NetworkBase {
  EmbeddingNet<activation> pokemon_net;
  EmbeddingNet<activation> active_net;

  using T = std::conditional_t<discrete, uint8_t, float>;
  using Main = std::conditional_t<discrete, void, MainNet>;

  BattleCaches<T> battle_cache;

  uint32_t pokemon_out_dim;
  uint32_t active_out_dim;
  uint32_t side_embedding_dim;
  std::vector<T> battle_embedding;

public:
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

  float value_inference(const pkmn_gen1_battle &b,
                        const pkmn_gen1_chance_durations &d) {
    float value;
    write_battle_embedding<float>(battle_embedding.data(), b, d);
    value = sigmoid(main_net.propagate(battle_embedding.data()));
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

private:
  auto side_embedding_index(auto i) const noexcept {
    assert(i > 0);
    return (1 + active_out_dim) + (i - 1) * (1 + pokemon_out_dim);
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

namespace Impl {
std::shared_ptr<Network> invalid(const std::string &msg) {
  throw std::runtime_error{"Invalid layer size for quantized net " + msg +
                           " (check code for valid sizes)."};
  return std::shared_ptr<Network>{nullptr};
}
template <int In, int H1> std::shared_ptr<Network> make_network_2(int h2) {
  if (H1 < h2) {
    throw std::runtime_error{"Quantized net must have decreasing layer size."};
    return std::shared_ptr<Network>{nullptr};
  }

  switch (h2) {
  case 32:
    return std::make_shared<MainNet<In, H1, 32>>();
  case 64:
    return std::make_shared<MainNet<In, H1, 64>>();
  default:
    return Impl::invalid("H2: " + std::to_string(h2));
  }
}
template <int In> std::shared_ptr<Network> make_network_1(int h1, int h2) {
  switch (h1) {
  case 32:
    return make_network_2<In, 32>(h2);
  case 64:
    return make_network_2<In, 64>(h2);
  case 128:
    return make_network_2<In, 128>(h2);
  default:
    return Impl::invalid("H1: " + std::to_string(h1));
  }
}
} // namespace Impl

std::shared_ptr<Network> make_network(int in, int h1, int h2) {
  switch (in) {
  case 512:
    return Impl::make_network_1<512>(h1, h2);
  case 768:
    return Impl::make_network_1<768>(h1, h2);
  case 1024:
  default:
    return Impl::invalid("Side dim: " + std::to_string(in));
  }
}

} // namespace NN::Battle
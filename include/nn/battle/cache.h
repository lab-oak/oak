#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <libpkmn/data/status.h>

#include <map>

namespace NN::Battle {

// dynamically sized embeddings use new T[] rather than vector
template <typename T, int dim>
using EmbeddingT = std::conditional_t<(dim > 0), std::array<T, dim>, T *>;

using PKMN::Data::Status;

template <typename T, int dim = 0> struct PokemonCache {

  // Encode does not have a dimension for no status
  static constexpr auto n_status = Encode::Battle::Status::n_dim + 1;
  // We only encode whether the move has pp, so 2^4 for the moveset
  static constexpr auto n_pp = 16;
  // For a stored pokemon, the move pp and status features are the only ones
  // that can change over the course of the game (hp is not a part of the input
  // to the pokemon embedding)
  static constexpr uint8_t n_embeddings = n_status * n_pp;
  // All status consditions that dont use sleep duration
  static constexpr std::array<Status, 8> status_array{
      Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
      Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};

  static constexpr bool is_dynamic{dim <= 0};
  static constexpr bool is_integral{std::is_integral_v<T>};
  using Embedding = EmbeddingT<T, dim>;
  using Key = uint8_t;

  uint32_t embedding_size;
  std::array<Embedding, n_embeddings> embeddings;
  std::vector<float> embedding;

  PokemonCache(int d = 0) : embedding_size{dim}, embedding{} {
    if constexpr (is_dynamic) {
      embedding_size = d;
      for (auto &embedding : embeddings) {
        embedding = new T[embedding_size];
      }
    }
    if constexpr (is_integral) {
      embedding.resize(embedding_size);
    }
  }

  PokemonCache(const PokemonCache &other) {
    if constexpr (is_dynamic) {
      embedding_size = other.embedding_size;
      for (auto i = 0; i < n_embeddings; ++i) {
        embeddings[i] = new T[embedding_size];
        const auto *source = other.embeddings[i];
        std::copy(source, source + embedding_size, embeddings[i]);
      }
    } else {
      embeddings = other.embeddings;
    }
    embedding = other.embedding;
  }

  ~PokemonCache() {
    if constexpr (is_dynamic) {
      for (auto &embedding : embeddings) {
        delete[] embedding;
      }
    }
  }

  inline T *data(Key key) const {
    if constexpr (is_dynamic) {
      return embeddings[key];
    } else {
      return embeddings[key].data();
    }
  }

  // iterate through all move pp/status combinations for a pokemon and store
  // embedding
  void fill(auto &pokemon_net, const PKMN::Pokemon &base_pokemon) {
    assert(embedding_size == pokemon_net.fc1.out_dim);

    auto pokemon = base_pokemon;
    // iterate through all 16 has-pp combinations
    for (auto m = 0; m < n_pp; ++m) {

      // set move pp based on bits of m
      for (auto i = 0; i < 4; ++i) {
        pokemon.moves[i].pp = m & (1 << i);
      }

      const auto get_entry = [this, &pokemon_net](const auto &pokemon,
                                                  const auto sleep) {
        std::array<float, Encode::Battle::Pokemon::n_dim> encoding{};
        Encode::Battle::Pokemon::write(pokemon, sleep, encoding.data());
        auto *embedding_data =
            this->data(Encode::Battle::pokemon_key(pokemon, sleep));

        if constexpr (is_integral) {
          pokemon_net.propagate(encoding.data(), embedding.data());
          std::transform(embedding.begin(), embedding.end(), embedding_data,
                         [](const auto f) {
                           return static_cast<T>(std::numeric_limits<T>::max() *
                                                 f);
                         });
        } else {
          pokemon_net.propagate(encoding.data(), embedding_data);
        }
      };

      // non slept status conditions
      for (const auto status : status_array) {
        pokemon.status = status;
        get_entry(pokemon, 0);
      }

      // slept
      pokemon.status = Status::Sleep1;
      for (auto sleep = 1; sleep <= 7; ++sleep) {
        get_entry(pokemon, sleep);
      }
    }
  }

  const T *get(const auto &pokemon, const auto sleep) const {
    const auto key = Encode::Battle::pokemon_key(pokemon, sleep);
    return data(key);
  }
};

template <typename T, int dim = 0> struct ActivePokemonCache {

  static constexpr bool is_dynamic{dim <= 0};
  static constexpr bool is_integral{std::is_integral_v<T>};
  using Embedding = EmbeddingT<T, dim>;
  using Key = std::pair<PKMN::ActivePokemon, uint8_t>;

  uint32_t embedding_size;
  std::map<Key, Embedding> embeddings;
  // this is done at runtime for ActiveCache so it's a member
  std::array<float, Encode::Battle::Active::n_dim> encoding;
  std::vector<float> embedding;

  ActivePokemonCache(int d = 0) : embedding_size{dim} {
    if constexpr (is_dynamic) {
      embedding_size = d;
    }
    if constexpr (is_integral) {
      embedding.resize(embedding_size);
    }
  }

  ~ActivePokemonCache() {
    if constexpr (is_dynamic) {
      for (auto p : embeddings) {
        delete[] p.second;
      }
    }
  }

  ActivePokemonCache(const ActivePokemonCache &other) {
    if constexpr (is_dynamic) {
      embedding_size = other.embedding_size;
      for (const auto &p : other.embeddings) {
        embeddings[p.first] = new T[embedding_size];
        const auto *source = p.second;
        std::copy(source, source + embedding_size, embeddings[p.first]);
      }
    } else {
      embeddings = other.embeddings;
    }
  }

  auto *data(const auto key) {
    if constexpr (is_dynamic) {
      return embeddings[key];
    } else {
      return embeddings[key].data();
    }
  }

  // TODO
  void fill(auto &, const PKMN::Pokemon &) {}

  const T *get(auto &active_net, const auto &active, const auto &pokemon,
               const auto &duration) {
    const auto key = std::pair<PKMN::ActivePokemon, uint8_t>{
        active, Encode::Battle::pokemon_key(pokemon, duration.sleep(0))};
    if (embeddings.find(key) != embeddings.end()) {
      const auto embedding_data = data(key);
      assert(embedding_data != nullptr);
      return embedding_data;
    } else {
      std::fill(encoding.begin(), encoding.end(), 0);
      Encode::Battle::Active::write(pokemon, active, duration, encoding.data());

      if constexpr (is_dynamic) {
        embeddings[key] = new T[embedding_size];
      }
      auto *embedding_data = data(key);

      if constexpr (is_integral) {
        active_net.propagate(encoding.data(), embedding.data());
        std::transform(embedding.begin(), embedding.end(), embedding_data,
                       [](const auto f) {
                         return static_cast<T>(std::numeric_limits<T>::max() *
                                               f);
                       });
      } else {
        active_net.propagate(encoding.data(), embedding_data);
      }

      return embedding_data;
    }
  }
};

} // namespace NN::Battle

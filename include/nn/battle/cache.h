#pragma once

#include <encode/battle/battle.h>
#include <encode/battle/key.h>
#include <libpkmn/data/status.h>
#include <nn/battle/cache.h>

namespace NN {

using PKMN::Data::Status;

template <typename T, int dim = 39> struct PokemonCache {

  // Encode does not have a dimension for no status
  static constexpr auto n_status = Encode::Status::n_dim + 1;
  // We only encode whether the move has pp, so 2^4 for the moveset
  static constexpr auto n_pp = 16;
  // For a stored pokemon, the move pp and status features are the only ones
  // that can change over the course of the game (hp is not a part of the input
  // to the pokemon embedding)
  static constexpr uint8_t n_embeddings = n_status * n_pp;

  // all status consditions that dont use sleep duration
  static constexpr std::array<Status, 9> status_array{
      Status::None,      Status::Poison, Status::Burn,  Status::Freeze,
      Status::Paralysis, Status::Rest1,  Status::Rest2, Status::Rest3};

  using Embedding = std::array<T, dim>;
  std::array<Embedding, n_embeddings> embeddings;

  // iterate through all move pp/status combinations for a pokemon and store
  // embedding
  void fill(const auto &pokemon_net, const PKMN::Pokemon &base_pokemon) {
    auto pokemon = base_pokemon;
    // iterate through all 16 has-pp combinations
    for (auto m = 0; m < 16; ++m) {

      // set move pp based on bits of m
      for (auto i = 0; i < 4; ++i) {
        pokemon.moves[i].pp = m & (1 << i);
      }

      const auto get_entry = [&pokemon_net, &pokemon, this](const auto sleep) {
        std::array<float, Encode::Pokemon::n_dim> input{};
        Encode::Pokemon::write(pokemon, sleep, input.data());
        const auto key = Encode::Battle::pokemon_key(pokemon, sleep);
        pokemon_net.propagate(input.data(), this->embeddings[key].data());
      };

      // non slept status conditions
      for (const auto status : status_array) {
        pokemon.status = status;
        get_entry(0);
      }

      // slept
      pokemon.status = Status::Sleep1;
      for (auto sleep = 1; sleep <= 7; ++sleep) {
        get_entry(sleep);
      }
    }
  }

  const T *get(const auto &pokemon, const auto sleep) const {
    const auto key = Encode::Battle::pokemon_key(pokemon, sleep);
    return embeddings[key].data();
  }
};

} // namespace NN
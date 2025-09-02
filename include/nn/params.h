#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>

namespace NN {

namespace Battle {
static constexpr int pokemon_hidden_dim = 64;
static constexpr int pokemon_out_dim = 39;
static constexpr int active_hidden_dim = 64;
static constexpr int active_out_dim = 55;
static constexpr int side_out_dim =
    (1 + active_out_dim) + 5 * (1 + pokemon_out_dim);
static constexpr int hidden_dim = 128;
static constexpr int value_hidden_dim = 32;
static constexpr int policy_hidden_dim = 32;
static constexpr int policy_out_dim =
    static_cast<int>(PKMN::Data::Species::Mew) +
    (static_cast<int>(PKMN::Data::Move::Struggle) - 1); // no Struggle, None
} // namespace Battle

namespace Build {
static constexpr int policy_hidden_dim = 512;
static constexpr int value_hidden_dim = 256;
} // namespace Build

} // namespace NN
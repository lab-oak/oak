#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Battle {

namespace py = pybind11;

namespace {

struct Output {
  size_t size;
  size_t pokemon_out_dim;
  size_t active_out_dim;
  size_t side_out_dim;
  py::array_t<float> pokemon;
  py::array_t<float> active_pokemon;
  py::array_t<float> sides;
  py::array_t<float> value;
  py::array_t<float> policy_logit;
  py::array_t<float> policy;

  // last dim is neg inf, invalid actions map to it
  static constexpr size_t policy_out_dim = Encode::Battle::Policy::n_dim + 1;

  Output(size_t size, size_t pod = NN::Battle::Default::pokemon_out_dim,
         size_t aod = NN::Battle::Default::active_out_dim)
      : size{size}, pokemon_out_dim{pod}, active_out_dim{aod},
        side_out_dim{(1 + active_out_dim) + 5 * (1 + pokemon_out_dim)} {
    pokemon =
        py::array_t<float>(std::vector<size_t>{size, 2, 5, pokemon_out_dim});
    active_pokemon =
        py::array_t<float>(std::vector<size_t>{size, 2, 1, active_out_dim});
    sides = py::array_t<float>(std::vector<size_t>{size, 2, 1, side_out_dim});
    value = py::array_t<float>(std::vector<size_t>{size, 1});
    policy_logit =
        py::array_t<float>(std::vector<size_t>{size, 2, policy_out_dim});
    policy = py::array_t<float>(std::vector<size_t>{size, 2, 9});
  }

  void clear() {
    std::fill_n(pokemon.mutable_data(), pokemon.size(), 0.0f);
    std::fill_n(active_pokemon.mutable_data(), active_pokemon.size(), 0.0f);
    std::fill_n(sides.mutable_data(), sides.size(), 0.0f);
    std::fill_n(value.mutable_data(), value.size(), 0.0f);
    std::fill_n(policy_logit.mutable_data(), policy_logit.size(), 0.0f);
    std::fill_n(policy.mutable_data(), policy.size(), 0.0f);
    auto logit = policy_logit.mutable_unchecked<3>();
    for (auto s = 0; s < 2; ++s) {
      for (size_t i = 0; i < logit.shape(0); ++i) {
        logit(i, s, logit.shape(2) - 1) =
            -std::numeric_limits<float>::infinity();
      }
    }
  }
};

} // namespace

} // namespace Py::Battle
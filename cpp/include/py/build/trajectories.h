#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Build {

namespace py = pybind11;

using Tensorizer = Encode::Build::Tensorizer<>;

namespace {

struct Trajectories {
  size_t size;
  py::array_t<int64_t> actions;
  py::array_t<int64_t> mask;
  py::array_t<float> policy;
  py::array_t<float> value;
  py::array_t<float> score;
  py::array_t<int64_t> start;
  py::array_t<int64_t> end;

  Trajectories(size_t size) : size{size} {
    actions = py::array_t<int64_t>(std::vector<size_t>{size, 31, 1});
    mask = py::array_t<int64_t>(
        std::vector<size_t>{size, 31, Tensorizer::max_actions});
    start = py::array_t<int64_t>(std::vector<size_t>{size, 1});
    end = py::array_t<int64_t>(std::vector<size_t>{size, 1});
    policy = py::array_t<float>(std::vector<size_t>{size, 31, 1});
    value = py::array_t<float>(std::vector<size_t>{size, 1});
    score = py::array_t<float>(std::vector<size_t>{size, 1});
  }

  void clear() {
    std::fill_n(actions.mutable_data(), actions.size(), int64_t{});
    std::fill_n(mask.mutable_data(), mask.size(), int64_t{});
    std::fill_n(start.mutable_data(), start.size(), int64_t{});
    std::fill_n(end.mutable_data(), end.size(), int64_t{});
    std::fill_n(policy.mutable_data(), policy.size(), float{});
    std::fill_n(value.mutable_data(), value.size(), float{});
    std::fill_n(score.mutable_data(), score.size(), float{});
  }
};

// // These two structs store what is *missing* so they can quickly write the
// // actions masks
// template <typename F = Format::OU> struct SetHelper {
//   Species species;
//   std::array<Move, F::max_move_pool_size> move_pool;
//   uint32_t n_moves;
//   uint32_t move_pool_size;

//   SetHelper() = default;
//   SetHelper(const auto species)
//       : species{static_cast<Species>(species)}, n_moves{},
//         move_pool_size{F::move_pool_size(species)},
//         move_pool{F::move_pool(species)} {}

//   auto begin() { return move_pool.begin(); }
//   const auto begin() const { return move_pool.begin(); }
//   auto end() { return move_pool.begin() + move_pool_size; }
//   const auto end() const { return move_pool.begin() + move_pool_size; }

//   void add_move(const auto m) {
//     const auto move_pool_end =
//         std::remove(begin(), end(), static_cast<Move>(m));
//     if (move_pool_end != end()) {
//       --move_pool_size;
//       ++n_moves;
//     }
//   }

//   bool complete() const { return (n_moves >= 4) || (move_pool_size == 0); }
// };

// template <typename F = Format::OU> struct TeamHelper {
//   TeamHelper() : sets{}, size{} {
//     available_species = {F::legal_species.begin(), F::legal_species.end()};
//   }

//   std::array<SetHelper<F>, 6> sets;
//   int size;
//   std::vector<Species> available_species;

//   auto begin() { return sets.begin(); }
//   const auto begin() const { return sets.begin(); }
//   auto end() { return sets.begin() + size; }
//   const auto end() const { return sets.begin() + size; }

//   void apply_action(const auto action) {
//     const auto [s, m] = Tensorizer<F>::species_move_list(action);
//     const auto species = static_cast<Species>(s);
//     const auto move = static_cast<Move>(m);
//     if (move == Move::None) {
//       sets[size++] = SetHelper{species};
//       std::erase(available_species, species);
//     } else {
//       auto it = std::find_if(begin(), end(), [species](const auto &set) {
//         return set.species == species;
//       });
//       assert(it != end());
//       auto &set = *it;
//       set.add_move(move);
//     }
//   }

//   auto write_moves(auto *mask) const {
//     for (const auto &set : (*this)) {
//       if (!set.complete()) {
//         for (const auto move : set) {
//           *mask++ = Tensorizer<F>::species_move_table(set.species, move);
//         }
//       }
//     }
//     return mask;
//   }

//   auto write_species(auto *mask) {
//     for (const auto species : available_species) {
//       *mask++ = Tensorizer<F>::species_move_table(species, Move::None);
//     }
//     return mask;
//   }

//   auto write_swaps(auto *mask) const {
//     for (const auto &set : (*this)) {
//       *mask++ = Tensorizer<F>::species_move_table(set.species, Move::None);
//     }
//     return mask;
//   }
// };

// template <typename F = Format::OU> struct TrajectoryInput {
//   int64_t *action;
//   int64_t *mask;
//   float *policy;
//   float *value;
//   float *score;
//   int64_t *start;
//   int64_t *end;

//   TrajectoryInput index(auto i) const {
//     auto copy = *this;
//     copy.action += i * 31;
//     copy.mask += i * 31 * Tensorizer<F>::max_actions;
//     copy.policy += i * 31;
//     copy.value += i;
//     copy.score += i;
//     copy.start += i;
//     copy.end += i;
//     return copy;
//   }

//   void write(const CompressedTrajectory<F> &traj) {
//     constexpr float den = std::numeric_limits<uint16_t>::max();

//     TeamHelper<F> helper{};

//     // get bounds for mask logic
//     auto start = 0;
//     auto full = 0; // turn off species picks
//     auto swap = 0;
//     auto end = 0;
//     // find start
//     for (auto i = 0; i < 31; ++i) {
//       const auto update = traj.updates[i];
//       if (update.probability != 0) {
//         start = i;
//         break;
//       }
//     }
//     // notice the reversed for loop. find end, swap, full
//     for (auto i = 30; i >= 0; --i) {
//       const auto update = traj.updates[i];
//       if (update.probability != 0) {
//         const auto [s, m] = Tensorizer<F>::species_move_list(update.action);
//         if (end == 0) {
//           // + 1 because we use these like '< end' later
//           end = i + 1;
//           // is last move a species pick? (e.g. team size > 1)
//           // must be a swap
//           swap = end - (m == Move::None);
//         } else {
//           // need to find the last actual addition
//           // the last species add is always before the swap/end
//           // because movesets must be maximal
//           if (m == Move::None) {
//             full = i + 1;
//             break;
//           }
//         }
//       }
//     }

//     for (auto i = 0; i < 31; ++i) {
//       const auto &update = traj.updates[i];
//       auto mask_begin = mask;
//       auto mask_end = mask + Tensorizer<F>::max_actions;
//       if (i < start) {
//         *action++ = -1;
//         *policy++ = 0;
//         std::fill(mask_begin, mask_end, -1);
//         helper.apply_action(update.action);
//       } else {
//         // started
//         if (i < end) {
//           *action++ = update.action;
//           *policy++ = update.probability / den;
//           if (i < swap) {
//             if (i < full) {
//               mask = helper.write_species(mask);
//             }
//             mask = helper.write_moves(mask);
//           } else {
//             mask = helper.write_swaps(mask);
//           }
//           std::fill(mask, mask_end, -1);
//           auto chosen = std::find(mask_begin, mask, update.action);
//           std::swap(*chosen, *mask_begin);
//           helper.apply_action(update.action);
//         } else {
//           // ended
//           *action++ = -1;
//           *policy++ = 0;
//           std::fill(mask, mask_end, -1);
//         }
//       }
//       mask = mask_end;
//     } // end update loop

//     *(this->start)++ = start;
//     *(this->end)++ = end;
//     *value++ = traj.header.value / den;
//     if (traj.header.score == std::numeric_limits<uint16_t>::max()) {
//       *score++ = -1;
//     } else {
//       *score++ = traj.header.score / 2.0;
//     }

//     return;
//   }
// };


} // namespace

} // namespace Py::Build
#pragma once

#include <pkmn.h>

#include <battle/strings.h>
#include <libpkmn/data/layout.h>
#include <libpkmn/data/options.h>
#include <search/tree.h>
#include <search/durations.h>
#include <util/random.h>

#include <chrono>
#include <iostream>
#include <type_traits>

#include "../extern/lrsnash/src/lib.h"

struct BattleData {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

namespace MonteCarlo {
struct Model {
  prng device;
};
} // namespace MonteCarlo

struct MCTS {

  template <bool _root_matrix = true, size_t _root_rolls = 3,
            size_t _other_rolls = 1, bool _debug_print = false>
  struct Options {
    static constexpr bool root_matrix = _root_matrix;
    static constexpr size_t root_rolls = _root_rolls;
    static constexpr size_t other_rolls = _other_rolls;
    static constexpr bool debug_print = _debug_print;
    // secondary
    static constexpr bool rolls_same = (root_rolls == other_rolls);
    static constexpr bool clamping = (root_rolls != 39) || (other_rolls != 39);
  };

  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  size_t total_nodes;
  size_t total_depth;
  std::array<pkmn_choice, 9> choices;
  std::array<std::array<uint32_t, 9>, 9> visit_matrix;
  std::array<std::array<float, 9>, 9> value_matrix;

  struct Output {
    uint m;
    uint n;
    size_t iterations;
    double total_value;
    double empirical_value;
    double nash_value;
    std::array<pkmn_choice, 9> p1_choices;
    std::array<pkmn_choice, 9> p2_choices;
    std::array<double, 9> p1_empirical;
    std::array<double, 9> p2_empirical;
    std::array<double, 9> p1_nash;
    std::array<double, 9> p2_nash;
    std::array<std::array<size_t, 9>, 9> visit_matrix;
    std::array<std::array<double, 9>, 9> value_matrix;
    std::chrono::milliseconds duration;
  };

  template <typename Options = Options<>>
  auto run(const auto dur, auto &node, const auto &input, auto &model,
           Output output = {}) {

    *this = {};
    if (!node.is_init()) {
      auto m = pkmn_gen1_battle_choices(&input.battle, PKMN_PLAYER_P1,
                                        pkmn_result_p1(input.result),
                                        choices.data(), PKMN_GEN1_MAX_CHOICES);
      auto n = pkmn_gen1_battle_choices(&input.battle, PKMN_PLAYER_P2,
                                        pkmn_result_p2(input.result),
                                        choices.data(), PKMN_GEN1_MAX_CHOICES);
      node.init(m, n);
    }

    const auto prepare_and_run_iteration = [this, &input, &model,
                                            &node]() -> float {
      auto copy = input;
      std::bit_cast<uint64_t *>(copy.battle.bytes +
                                Layout::Offsets::Battle::rng)[0] =
          model.device.uniform_64();
      chance_options.durations = copy.durations;
      apply_durations(copy.battle, copy.durations);
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
      return run_iteration<Options>(&node, copy, model).first;
    };

    // time duration
    if constexpr (requires {
                    std::chrono::duration_cast<std::chrono::milliseconds>(dur);
                  }) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
      std::chrono::milliseconds elapsed{};
      while (elapsed < duration) {
        output.total_value += prepare_and_run_iteration();
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      output.duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
      // flag duration
    } else if constexpr (requires { *dur; }) {
      const auto start = std::chrono::high_resolution_clock::now();
      while (*dur) {
        ++output.iterations;
        output.total_value += prepare_and_run_iteration();
      }
      const auto end = std::chrono::high_resolution_clock::now();
      output.duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      // iteration count
    } else {
      const auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < dur; ++i) {
        output.total_value += prepare_and_run_iteration();
      }
      output.iterations += dur;
      const auto end = std::chrono::high_resolution_clock::now();
      output.duration +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    // process output

    output.empirical_value = output.total_value / output.iterations;
    output.m = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P1, pkmn_result_p1(input.result),
        output.p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
    output.n = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P2, pkmn_result_p2(input.result),
        output.p2_choices.data(), PKMN_GEN1_MAX_CHOICES);

    if constexpr (Options::root_matrix) {
      // emprically shown to be reliable for 9x9 matrices, 128 bit lrslib
      constexpr int discretize_factor = 80;
      std::array<int, 9 * 9> solve_matrix;

      for (int i = 0; i < output.m; ++i) {
        for (int j = 0; j < output.n; ++j) {
          output.visit_matrix[i][j] += visit_matrix[i][j];
          output.value_matrix[i][j] += value_matrix[i][j];
          auto n = output.visit_matrix[i][j];
          output.p1_empirical[i] += n;
          output.p2_empirical[j] += n;
          n += !n;
          solve_matrix[output.n * i + j] =
              output.value_matrix[i][j] / n * discretize_factor;
        }
      }

      LRSNash::FastInput solve_input{static_cast<int>(output.m),
                                     static_cast<int>(output.n),
                                     solve_matrix.data(), discretize_factor};
      // 2 extra entries for output denom, nash value
      std::array<float, 9 + 2> nash1{}, nash2{};
      LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
      bool success = true;
      try {
        LRSNash::solve_fast(&solve_input, &solve_output);
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        success = false;
        std::cout << "value matrix" << std::endl;
        for (auto i = 0; i < output.m; ++i) {
          for (auto j = 0; j < output.n; ++j) {
            auto n = output.visit_matrix[i][j];
            n += (n == 0);
            std::cout << output.value_matrix[i][j] / n << ' ';
          }
          std::cout << std::endl;
        }
      }

      for (int i = 0; i < output.m; ++i) {
        output.p1_empirical[i] /= (float)output.iterations;
        output.p1_nash[i] = nash1[i];
      }
      for (int j = 0; j < output.n; ++j) {
        output.p2_empirical[j] /= (float)output.iterations;
        output.p2_nash[j] = nash2[j];
      }

      if (!success) {
        output.p1_nash = output.p1_empirical;
        output.p2_nash = output.p2_empirical;
      }
      output.nash_value = solve_output.value;
    }

    return output;
  }

  template <size_t n_rolls>
  static constexpr uint8_t roll_byte(const uint8_t seed) {
    constexpr uint8_t lowest_roll{236};
    constexpr uint8_t middle_roll{236};
    if constexpr (n_rolls == 1) {
      return middle_roll;
    } else {
      static_assert((n_rolls == 2) || (n_rolls == 3) || (n_rolls == 20));
      constexpr uint8_t step = 38 / (n_rolls - 1);
      return lowest_roll + step * (seed % n_rolls);
    }
  }

  template <typename Options>
  std::pair<float, float> run_iteration(auto *node, auto &input, auto &model,
                                        size_t depth = 0) {

    auto &battle = input.battle;
    auto &durations = input.durations;
    auto &result = input.result;
    auto &device = model.device;

    const auto print = [depth](const auto &data, bool new_line = true) -> void {
      if constexpr (!Options::debug_print) {
        return;
      }
      for (auto i = 0; i < depth; ++i) {
        std::cout << "  ";
      }
      std::cout << data;
      if (new_line) {
        std::cout << '\n';
      }
    };

    const auto battle_options_set = [this, &battle, depth]() {
      if constexpr (!Options::clamping) {
        pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      } else {
        // last two bytes of battle rng
        const auto *rand = battle.bytes + Layout::Offsets::Battle::rng + 6;
        auto *over = this->calc_options.overrides.bytes;
        if constexpr (Options::rolls_same) {
          over[0] = roll_byte<Options::root_rolls>(rand[0]);
          over[8] = roll_byte<Options::root_rolls>(rand[1]);
        } else {

          if (depth == 0) {
            over[0] = roll_byte<Options::root_rolls>(rand[0]);
            over[8] = roll_byte<Options::root_rolls>(rand[1]);
          } else {
            over[0] = roll_byte<Options::other_rolls>(rand[0]);
            over[8] = roll_byte<Options::other_rolls>(rand[1]);
          }
        }
        pkmn_gen1_battle_options_set(&this->options, nullptr, nullptr,
                                     &this->calc_options);
      }
    };

    if (node->stats().is_init()) {
      using Bandit = std::remove_reference_t<decltype(node->stats())>;
      using Outcome = typename Bandit::Outcome;

      // do bandit
      Outcome outcome;

      node->stats().select(device, outcome);
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = choices[outcome.p1_index];
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[outcome.p2_index];

      print(
          "P1: " + Strings::side_choice_string(battle.bytes, c1) + " P2: " +
          Strings::side_choice_string(battle.bytes + Layout::Sizes::Side, c2));
      print(node->stats().visit_string());

      battle_options_set();
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
      const auto obs = std::bit_cast<const std::array<uint8_t, 16>>(
          *pkmn_gen1_battle_options_chance_actions(&options));

      auto *child = (*node)(outcome.p1_index, outcome.p2_index, obs);
      const auto value = run_iteration<Options>(child, input, model, depth + 1);
      outcome.p1_value = value.first;
      outcome.p2_value = value.second;
      node->stats().update(outcome);

      print("value: " + std::to_string(value.first));

      if constexpr (Options::root_matrix) {
        if (depth == 0) {
          ++visit_matrix[outcome.p1_index][outcome.p2_index];
          value_matrix[outcome.p1_index][outcome.p2_index] += value.first;
        }
      }

      return value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        print("Initializing node");
        ++total_nodes;
        if constexpr (requires {
                        model.inference(input.battle, input.durations);
                      }) {
          const auto m = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
              PKMN_GEN1_MAX_CHOICES);
          const auto n = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
              PKMN_GEN1_MAX_CHOICES);
          node->stats().init(m, n);
          const float value = model.inference(input.battle, input.durations);
          return {value, 1 - value};
        } else {
          return init_stats_and_rollout(node->stats(), device, battle, result);
        }
      }
    case PKMN_RESULT_WIN: {
      return {1, 0};
    }
    case PKMN_RESULT_LOSE: {
      return {0, 1};
    }
    case PKMN_RESULT_TIE: {
      return {.5, .5};
    }
    default: {
      assert(false);
      return {.5, .5};
    }
    };
  }

  std::pair<float, float> init_stats_and_rollout(auto &stats, auto &prng,
                                                 pkmn_gen1_battle &battle,
                                                 pkmn_result result) {

    auto seed = prng.uniform_64();
    auto m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                      pkmn_result_p1(result), choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    auto c1 = choices[seed % m];
    auto n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                      pkmn_result_p2(result), choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    auto c2 = choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    stats.init(m, n);
    while (!pkmn_result_type(result)) {
      seed = prng.uniform_64();
      m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                   pkmn_result_p1(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      c1 = choices[seed % m];
      n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                   pkmn_result_p2(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      c2 = choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    }
    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_WIN: {
      return {1, 0};
    }
    case PKMN_RESULT_LOSE: {
      return {0, 1};
    }
    case PKMN_RESULT_TIE: {
      return {.5, .5};
    }
    default: {
      assert(false);
      return {.5, .5};
    }
    };
  }
};
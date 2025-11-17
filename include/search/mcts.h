#pragma once

#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <search/durations.h>
#include <search/tree.h>
#include <util/random.h>

#include <chrono>
#include <iostream>
#include <type_traits>

#include "../extern/lrsnash/src/lib.h"

namespace MCTS {

struct BattleData {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

struct MonteCarlo {
  mt19937 device;
};

// strategies, value estimate, etc. All info the search produces
struct Output {
  uint8_t m;
  uint8_t n;
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

using Obs = std::array<uint8_t, 16>;

template <bool _root_matrix = true, bool _root_matrix_ucb = true,
          size_t _root_matrix_ucb_delay = (1ULL << 12),
          size_t _root_matrix_ucb_interval = (1ULL << 6),
          size_t _root_rolls = 3, size_t _other_rolls = 1,
          bool _debug_print = false>
struct SearchOptions {
  static constexpr bool root_matrix = _root_matrix;
  static constexpr bool root_matrix_ucb = _root_matrix_ucb;
  static constexpr size_t root_matrix_ucb_delay = _root_matrix_ucb_delay;
  static constexpr size_t root_matrix_ucb_interval = _root_matrix_ucb_interval;
  static constexpr size_t root_rolls = _root_rolls;
  static constexpr size_t other_rolls = _other_rolls;
  static constexpr bool debug_print = _debug_print;
  // secondary
  static constexpr bool rolls_same = (root_rolls == other_rolls);
  static constexpr bool clamping = (root_rolls != 39) || (other_rolls != 39);
};

template <typename Options = SearchOptions<>> struct Search {

  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  std::array<std::array<uint32_t, 9>, 9> visit_matrix;
  std::array<std::array<float, 9>, 9> value_matrix;
  // matrix ucb
  float ucb_weight;
  std::array<float, 9 + 2> p1_nash;
  std::array<float, 9 + 2> p2_nash;

  Output run(const auto dur, const auto &bandit_params, auto &node, auto &model,
             const MCTS::BattleData &input, Output output = {}) {

    // reset data members
    *this = {};

    // get choices data here for matrix ucb
    output.m = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P1, pkmn_result_p1(input.result),
        output.p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
    output.n = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P2, pkmn_result_p2(input.result),
        output.p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
    if constexpr (Options::root_matrix_ucb) {
      ucb_weight = std::log(2 * output.m * output.n);
    }

    if (!node.is_init()) {

      node.init(output.m, output.n);

      if constexpr (requires {
                      node.stats().softmax_logits(nullptr, nullptr);
                      model.inference(
                          input.battle,
                          *pkmn_gen1_battle_options_chance_durations(&options));
                    }) {
        static thread_local std::array<float, 9> p1_logits;
        static thread_local std::array<float, 9> p2_logits;
        const float value = model.inference(
            input.battle, *pkmn_gen1_battle_options_chance_durations(&options),
            output.m, output.n, output.p1_choices.data(),
            output.p2_choices.data(), p1_logits.data(), p2_logits.data());
        node.stats().softmax_logits(p1_logits.data(), p2_logits.data());
      }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    // Three types for 'dur' (the time investement of the search) are allowed:
    // chrono duration
    if constexpr (requires {
                    std::chrono::duration_cast<std::chrono::milliseconds>(dur);
                  }) {
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
      std::chrono::milliseconds elapsed{};
      while (elapsed < duration) {
        output.total_value +=
            run_root_iteration(bandit_params, node, input, model, output);
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      // run while boolean flag is set
    } else if constexpr (requires { *dur; }) {
      while (*dur) {
        ++output.iterations;
        output.total_value +=
            run_root_iteration(bandit_params, node, input, model, output);
      }
      // number of iterations
    } else {
      for (auto i = 0; i < dur; ++i) {
        output.total_value +=
            run_root_iteration(bandit_params, node, input, model, output);
        ++output.iterations;
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    output.duration +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return process_output(output);
  }

  float run_root_iteration(const auto &bandit_params, auto &node,
                           const auto &input, auto &model, Output &output) {
    auto copy = input;
    auto *rng = reinterpret_cast<uint64_t *>(
        copy.battle.bytes + PKMN::Layout::Offsets::Battle::rng);
    rng[0] = model.device.uniform_64();
    chance_options.durations = copy.durations;
    apply_durations(copy.battle, copy.durations);
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

    if constexpr (Options::root_matrix_ucb) {
      if ((output.iterations >= Options::root_matrix_ucb_delay)) {

        uint8_t p1_index{}, p2_index{};

        if (output.iterations % Options::root_matrix_ucb_interval == 0) {

          // get ucb matrices

          std::array<int, 9 * 9> p1_ucb_matrix;
          std::array<int, 9 * 9> p2_ucb_matrix;
          constexpr int discretize_factor = 256;

          const float log_T = std::log(output.iterations);

          for (auto i = 0; i < output.m; ++i) {
            for (auto j = 0; j < output.n; ++j) {

              float p1_entry = 0;
              float p2_entry = 0;

              if (visit_matrix[i][j] > 0) {
                p1_entry = value_matrix[i][j] / visit_matrix[i][j];
                p2_entry = 1 - p1_entry;
              }

              const float exploration = std::sqrt(2 * (2 * log_T + ucb_weight) /
                                                  (visit_matrix[i][j] + 1));

              p1_entry += exploration;
              p2_entry += exploration;

              p1_ucb_matrix[i * output.n + j] = p1_entry * discretize_factor;
              p2_ucb_matrix[i * output.n + j] = p2_entry * discretize_factor;

            }
            // std::cout << std::endl;
          }

          // solve and sample
          std::array<float, 9 + 2> dummy;
          {
            LRSNash::FastInput solve_input{
                static_cast<int>(output.m), static_cast<int>(output.n),
                p1_ucb_matrix.data(), discretize_factor};
            LRSNash::FloatOneSumOutput solve_output{p1_nash.data(),
                                                    dummy.data(), 0};
            LRSNash::solve_fast(&solve_input, &solve_output);
          }

          {
            LRSNash::FastInput solve_input{
                static_cast<int>(output.m), static_cast<int>(output.n),
                p2_ucb_matrix.data(), discretize_factor};
            LRSNash::FloatOneSumOutput solve_output{dummy.data(),
                                                    p2_nash.data(), 0};
            LRSNash::solve_fast(&solve_input, &solve_output);
          }

          for (auto i = 0; i < output.m; ++i) {
            std::cout << int{100000 * p1_nash[i]} / 100 << ' ';
          }
          std::cout << std::endl;
          for (auto i = 0; i < output.n; ++i) {
            std::cout << int{100000 * p2_nash[i]} / 100 << ' ';
          }
          std::cout << std::endl;
          std::cout << std::endl;
        }

        {
          float p = model.device.uniform();
          for (auto i = 0; i < output.m; ++i) {
            p -= p1_nash[i];
            if (p <= 0) {
              p1_index = i;
              break;
            }
          }
        }

        {
          float p = model.device.uniform();
          for (auto i = 0; i < output.n; ++i) {
            p -= p2_nash[i];
            if (p <= 0) {
              p2_index = i;
              break;
            }
          }
        }

        auto c1 = p1_choices[p1_index];
        auto c2 = p1_choices[p2_index];
        battle_options_set(copy.battle, 0);
        copy.result = pkmn_gen1_battle_update(&copy.battle, c1, c2, &options);
        const auto obs = std::bit_cast<const Obs>(
            *pkmn_gen1_battle_options_chance_actions(&options));
        auto &child = node(p1_index, p2_index, obs);

        const auto value =
            run_iteration(bandit_params, child, copy, model, 1).first;

        ++visit_matrix[p1_index][p2_index];
        value_matrix[p1_index][p2_index] += value;

        return value;
      } else {
        return run_iteration(bandit_params, node, copy, model).first;
      }
    } else {
      return run_iteration(bandit_params, node, copy, model).first;
    }
  }

  // typical recursive mcts function
  // we return value for each player because it's slightly faster than calcing 1
  // - value at each node
  std::pair<float, float> run_iteration(const auto &bandit_params, auto &node,
                                        auto &input, auto &model,
                                        size_t depth = 0) {

    auto &battle = input.battle;
    auto &durations = input.durations;
    auto &result = input.result;
    auto &device = model.device;

    // debug print, optimized out if not enabled
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

    if (node.stats().is_init()) {
      using Bandit = std::remove_reference_t<decltype(node.stats())>;
      using JointOutcome = typename Bandit::JointOutcome;

      // do bandit
      JointOutcome outcome;

      node.stats().select(device, bandit_params, outcome);
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = p1_choices[outcome.p1.index];
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = p2_choices[outcome.p2.index];

      print("P1: " + PKMN::side_choice_string(battle.bytes, c1) + " P2: " +
            PKMN::side_choice_string(battle.bytes + PKMN::Layout::Sizes::Side,
                                     c2));

      battle_options_set(battle, depth);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
      const auto obs = std::bit_cast<const Obs>(
          *pkmn_gen1_battle_options_chance_actions(&options));

      auto &child = node(outcome.p1.index, outcome.p2.index, obs);
      const auto value =
          run_iteration(bandit_params, child, input, model, depth + 1);
      outcome.p1.value = value.first;
      outcome.p2.value = value.second;
      node.stats().update(outcome);

      print("value: " + std::to_string(value.first));

      if constexpr (Options::root_matrix) {
        if (depth == 0) {
          ++visit_matrix[outcome.p1.index][outcome.p2.index];
          value_matrix[outcome.p1.index][outcome.p2.index] += value.first;
        }
      }

      return value;
    }

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        print("Initializing node");
        // model is a net
        float value;
        if constexpr (requires {
                        model.inference(input.battle, input.durations);
                      }) {
          const auto m = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
              p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
          const auto n = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
              p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
          node.stats().init(m, n);
          if constexpr (requires {
                          node.stats().softmax_logits(nullptr, nullptr);
                        }) {
            static thread_local std::array<float, 9> p1_logits;
            static thread_local std::array<float, 9> p2_logits;
            value = model.inference(
                input.battle,
                *pkmn_gen1_battle_options_chance_durations(&options), m, n,
                p1_choices.data(), p2_choices.data(), p1_logits.data(),
                p2_logits.data());
            node.stats().softmax_logits(p1_logits.data(), p2_logits.data());
          } else {
            value = model.inference(
                input.battle,
                *pkmn_gen1_battle_options_chance_durations(&options));
          }
          return {value, 1 - value};
        } else {
          // model is monte-carlo
          return init_stats_and_rollout(node.stats(), device, battle, result);
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

  std::pair<float, float> init_stats_and_rollout(auto &stats, auto &device,
                                                 pkmn_gen1_battle &battle,
                                                 pkmn_result result) {

    auto seed = device.uniform_64();
    auto m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                      pkmn_result_p1(result), p1_choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    auto c1 = p1_choices[seed % m];
    auto n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                      pkmn_result_p2(result), p2_choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    auto c2 = p2_choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    stats.init(m, n);
    while (!pkmn_result_type(result)) {
      seed = device.uniform_64();
      m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                   pkmn_result_p1(result), p1_choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      c1 = p1_choices[seed % m];
      n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                   pkmn_result_p2(result), p2_choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      c2 = p2_choices[seed % n];
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

  // use battle seed to quickly compute a clamped damage roll
  template <size_t n_rolls>
  static constexpr uint8_t roll_byte(const uint8_t seed) {
    constexpr uint8_t lowest_roll{217};
    constexpr uint8_t middle_roll{236};
    if constexpr (n_rolls == 1) {
      return middle_roll;
    } else {
      static_assert((n_rolls == 2) || (n_rolls == 3) || (n_rolls == 20));
      constexpr uint8_t step = 38 / (n_rolls - 1);
      return lowest_roll + step * (seed % n_rolls);
    }
  }

  // pkmn_gen1_battle_options_set with constexpr logic
  void battle_options_set(pkmn_gen1_battle &battle, size_t depth) {
    if constexpr (!Options::clamping) {
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    } else {
      // last two bytes of battle rng
      const auto *rand = battle.bytes + PKMN::Layout::Offsets::Battle::rng + 6;
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
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, &calc_options);
    }
  }

  Output process_output(Output &output) {

    // prepare output, solve empirical root matrix if enabled
    output.empirical_value = output.total_value / output.iterations;

    if constexpr (Options::root_matrix) {
      // emprically shown to be reliable for 9x9 matrices, 128 bit lrslib
      // TODO maybe use gmp since this is not the hot part of the code
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
      // LRSNash convention: 2 extra entries needed for output denom, nash value
      std::array<float, 9 + 2> nash1{}, nash2{};
      LRSNash::FloatOneSumOutput solve_output{nash1.data(), nash2.data(), 0};
      bool success = true;
      try {
        LRSNash::solve_fast(&solve_input, &solve_output);
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        success = false;
        // print offending matrix in lossless form
        std::cout << "value matrix" << std::endl;
        for (auto i = 0; i < output.m; ++i) {
          for (auto j = 0; j < output.n; ++j) {
            std::cerr << "(" << output.value_matrix[i][j] << ", "
                      << output.visit_matrix[i][j] << ") ";
          }
          std::cerr << std::endl;
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
      output.nash_value = solve_output.value;

      // set nash data to emprical rather than leave as garbage/zeros/etc
      if (!success) {
        output.p1_nash = output.p1_empirical;
        output.p2_nash = output.p2_empirical;
        output.nash_value = output.empirical_value;
      }
    }

    return output;
  }
};

void print_output(const MCTS::Output &output, const pkmn_gen1_battle &battle,
                  const auto &p1_labels, const auto &p2_labels) {
  constexpr auto label_width = 8;

  auto print_arr = [](const auto &arr, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      std::cout << std::left << std::fixed << std::setw(label_width)
                << std::setprecision(3) << arr[i] << "  ";
    }
    std::cout << '\n';
  };

  const auto fix_label = [label_width](auto label) {
    std::stringstream ss{};
    ss << std::left << std::setw(label_width)
       << label.substr(0, label_width - 1);
    return ss.str();
  };

  std::cout << "Iterations: " << output.iterations
            << ", Time: " << output.duration.count() / 1000.0 << " sec\n";
  std::cout << "Value: " << std::fixed << std::setprecision(3)
            << output.empirical_value << "\n";

  std::cout << "\nP1" << std::endl;
  print_arr(p1_labels, output.m);
  print_arr(output.p1_empirical, output.m);
  print_arr(output.p1_nash, output.m);
  std::cout << "P2" << std::endl;
  print_arr(p2_labels, output.n);
  print_arr(output.p2_empirical, output.n);
  print_arr(output.p2_nash, output.n);

  std::cout << "\nMatrix:\n";
  std::array<char, label_width + 1> col_offset{};
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  std::cout << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.n; ++j)
    std::cout << fix_label(p2_labels[j]) << " ";
  std::cout << "\n";

  for (size_t i = 0; i < output.m; ++i) {
    std::cout << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.n; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        std::cout << " ----    ";
      } else {
        double avg = output.value_matrix[i][j] / output.visit_matrix[i][j];
        std::cout << std::left << std::fixed << std::setw(label_width)
                  << std::setprecision(3) << avg << " ";
      }
    }
    std::cout << '\n';
  }

  std::cout << "\nVisits:\n";
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  std::cout << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.n; ++j)
    std::cout << fix_label(p2_labels[j]) << " ";
  std::cout << "\n";

  for (size_t i = 0; i < output.m; ++i) {
    std::cout << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.n; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        std::cout << " ----    ";
      } else {
        auto avg = output.visit_matrix[i][j];
        std::cout << std::left << std::fixed << std::setw(label_width)
                  << std::setprecision(3) << avg << " ";
      }
    }
    std::cout << '\n';
  }
}

} // namespace MCTS
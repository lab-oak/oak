#pragma once

#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <search/durations.h>
#include <search/hash.h>
#include <util/random.h>

#include <chrono>
#include <iostream>
#include <type_traits>

#include "../extern/lrsnash/src/lib.h"

namespace TypeTraits {
template <typename T>
inline constexpr bool is_node =
    requires(std::remove_cvref_t<T> &heap) { heap.stats; };

template <typename T>
inline constexpr bool is_table =
    requires(std::remove_cvref_t<T> &heap) { heap.entries; };

template <typename T>
inline constexpr bool is_network = requires(std::remove_cvref_t<T> &network) {
  network.inference(std::declval<const pkmn_gen1_battle &>(),
                    std::declval<const pkmn_gen1_chance_durations &>());
};

template <typename T>
inline constexpr bool is_poke_engine =
    requires(std::remove_cvref_t<T> &poke_engine) {
      poke_engine.evaluate(std::declval<const pkmn_gen1_battle &>());
    };

template <typename T>
inline constexpr bool is_policy_network =
    requires(std::remove_cvref_t<T> &network) {
      network.inference(std::declval<const pkmn_gen1_battle &>(),
                        std::declval<const pkmn_gen1_chance_durations &>(),
                        std::declval<uint8_t>(), std::declval<uint8_t>(),
                        std::declval<const pkmn_choice *>(),
                        std::declval<const pkmn_choice *>(),
                        std::declval<float *>(), std::declval<float *>());
    };

template <typename T>
inline constexpr bool is_contextual_bandit =
    requires(std::remove_cvref_t<T> &stats) {
      stats.softmax_logits(
          std::declval<typename std::remove_cvref_t<T>::Params>(),
          std::declval<const float *>(), std::declval<const float *>());
    };

template <typename T>
inline constexpr bool is_matrix_ucb =
    requires(std::remove_cvref_t<T> &params) { params.bandit_params; };
} // namespace TypeTraits

namespace MCTS {
using namespace TypeTraits;

struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

// strategies, value estimate, etc. All info the search produces
struct Output {
  uint8_t m;
  uint8_t n;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;

  std::array<std::array<size_t, 9>, 9> visit_matrix;
  std::array<std::array<double, 9>, 9> value_matrix;

  size_t iterations;
  std::chrono::milliseconds duration;

  double empirical_value;
  double nash_value;
  std::array<double, 9> p1_prior;
  std::array<double, 9> p2_prior;
  std::array<double, 9> p1_empirical;
  std::array<double, 9> p2_empirical;
  std::array<double, 9> p1_nash;
  std::array<double, 9> p2_nash;
};

// for std::map compatibility
using Obs = std::array<uint8_t, 16>;

template <typename JointBandit> struct Node {
  using Key = std::tuple<uint8_t, uint8_t, Obs>;
  JointBandit stats;
  std::map<Key, Node<JointBandit>> children;
};

template <typename JointBandit> struct Table {
  using Key = uint64_t;
  Hash::Battle hasher;
  std::unordered_map<Key, JointBandit> entries;
};

size_t count(const auto &heap) {
  size_t c = 1;
  for (const auto &pair : heap.children) {
    c += count(pair.second);
  }
  return c;
}

struct MonteCarlo {};

// wrapper to use for enabling matrix ucb at root heap
template <typename BanditParams> struct MatrixUCBParams {
  BanditParams bandit_params;
  uint32_t delay;
  uint32_t interval;
  uint32_t minimum;
  float c;
};

template <size_t _root_rolls = 3, size_t _other_rolls = 1,
          bool _debug_print = false>
struct SearchOptions {
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
  Hash::State root_hash_state;

  // matrix ucb
  bool initial_solve;
  float ucb_weight;
  std::array<float, 9 + 2> p1_nash;
  std::array<float, 9 + 2> p2_nash;

  size_t total_depth;
  size_t errors;

  Output run(auto &device, const auto budget, const auto &params, auto &heap,
             auto &model, const Input &input, Output output = {}) {

    // reset data members
    *this = {};

    // get choices data here for matrix ucb
    output.m = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P1, pkmn_result_p1(input.result),
        output.p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
    output.n = pkmn_gen1_battle_choices(
        &input.battle, PKMN_PLAYER_P2, pkmn_result_p2(input.result),
        output.p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
    if constexpr (requires { params.bandit_params; }) {
      ucb_weight = std::log(2 * output.m * output.n);
    }

    if constexpr (is_poke_engine<decltype(model)>) {
      model.get_root_score(input.battle);
    }

    auto &stats = [&]() -> auto & {
      if constexpr (is_node<decltype(heap)>) {
        return heap.stats;
      } else {
        heap.hasher.init(input.battle, input.durations);
        root_hash_state = heap.hasher.state();
        return heap.entries[heap.hasher.last()];
      }
    }();

    if (!stats.is_init()) {

      stats.init(output.m, output.n);

      const auto bandit_params = [](const auto &params) -> const auto & {
        if constexpr (requires { params.bandit_params; }) {
          return params.bandit_params;
        } else {
          return params;
        }
      };

      if constexpr (is_contextual_bandit<decltype(stats)> &&
                    is_policy_network<decltype(model)>) {
        static thread_local std::array<float, 9> p1_logits;
        static thread_local std::array<float, 9> p2_logits;
        const float value =
            model.inference(input.battle, input.durations, output.m, output.n,
                            output.p1_choices.data(), output.p2_choices.data(),
                            p1_logits.data(), p2_logits.data());
        stats.softmax_logits(bandit_params(params), p1_logits.data(),
                             p2_logits.data());
      }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    // time duration
    if constexpr (requires {
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        budget);
                  }) {
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(budget);
      std::chrono::milliseconds elapsed{};
      while (elapsed < duration) {
        run_root_iteration(device, params, heap, input, model, output);
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      // run while boolean flag is set
    } else if constexpr (requires { *budget; }) {
      while (*budget) {
        run_root_iteration(device, params, heap, input, model, output);
        ++output.iterations;
      }
      // number of iterations
    } else {
      for (auto i = 0; i < budget; ++i) {
        run_root_iteration(device, params, heap, input, model, output);
        ++output.iterations;
      }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    output.duration +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    process_output(output);
    return output;
  }

  float run_root_iteration(auto &device, const auto &params, auto &heap,
                           const auto &input, auto &model, Output &output) {

    auto copy = input;
    auto *rng = reinterpret_cast<uint64_t *>(
        copy.battle.bytes + PKMN::Layout::Offsets::Battle::rng);
    rng[0] = device.uniform_64();
    chance_options.durations = copy.durations;
    randomize_hidden_variables(copy.battle, copy.durations);
    pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
    if constexpr (is_table<decltype(heap)>) {
      heap.hasher.set(root_hash_state);
    }

    if constexpr (!is_matrix_ucb<decltype(params)>) {
      return run_iteration(device, params, heap, copy, model, output).first;
    } else {
      if ((output.iterations < params.delay)) {
        return run_iteration(device, params.bandit_params, heap, copy, model,
                             output)
            .first;
      } else {
        // const auto start = std::chrono::high_resolution_clock::now();
        const auto [p1_index, p2_index] =
            solve_root_matrix_and_sample(device, params, copy, output);
        const auto c1 = output.p1_choices[p1_index];
        const auto c2 = output.p2_choices[p2_index];
        battle_options_set(copy.battle, 0);
        copy.result = pkmn_gen1_battle_update(&copy.battle, c1, c2, &options);

        const auto value = [&]() {
          if constexpr (is_node<decltype(heap)>) {
            const auto &obs = *reinterpret_cast<const Obs *>(
                pkmn_gen1_battle_options_chance_actions(&options));
            auto &child = heap.children[{p1_index, p2_index, obs}];
            return run_iteration(device, params.bandit_params, child, copy,
                                 model, output, 1);
          } else {
            return run_iteration(device, params.bandit_params, heap, copy,
                                 model, output, 1);
          }
        }();

        // TODO error check
        ++output.visit_matrix[p1_index][p2_index];
        output.value_matrix[p1_index][p2_index] += value.first;
        return value.first;
      }
    }
  }

  // typical recursive mcts function
  // we return value for each player because it's slightly faster than calcing 1
  // - value at each heap
  std::pair<float, float> run_iteration(auto &device, const auto &bandit_params,
                                        auto &heap, auto &input, auto &model,
                                        Output &output, size_t depth = 0) {
    static constexpr size_t max_depth = 100;

    bool error = false;
    if constexpr (is_table<decltype(heap)>) {
      if (depth >= max_depth) {
        set_turn_limit(input.battle);
        ++errors;
        error = true;
      }
    }

    auto &battle = input.battle;
    auto &result = input.result;
    auto &stats = [&]() -> auto & {
      if constexpr (is_node<decltype(heap)>) {
        return heap.stats;
      } else {
        return heap.entries[heap.hasher.last()];
      }
    }();

    if (stats.is_init() && !error) {
      using Bandit = std::remove_reference_t<decltype(stats)>;
      using JointOutcome = typename Bandit::JointOutcome;

      // do bandit
      JointOutcome outcome;

      stats.select(device, bandit_params, outcome);
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = p1_choices[outcome.p1.index];
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = p2_choices[outcome.p2.index];

      if constexpr (is_node<decltype(heap)>) {
        battle_options_set(battle, depth);
      } else {
        // battle_options_set(battle, depth);
        pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      }
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);

      if constexpr (is_table<decltype(heap)>) {
        heap.hasher.update(battle, durations(), c1, c2);
      }

      const auto value = [&]() {
        if constexpr (is_node<decltype(heap)>) {
          const auto &obs = *reinterpret_cast<const Obs *>(
              pkmn_gen1_battle_options_chance_actions(&options));
          auto &child =
              heap.children[{outcome.p1.index, outcome.p2.index, obs}];
          return run_iteration(device, bandit_params, child, input, model,
                               output, depth + 1);
        } else {
          return run_iteration(device, bandit_params, heap, input, model,
                               output, depth + 1);
        }
      }();
      outcome.p1.value = value.first;
      outcome.p2.value = value.second;

      if constexpr (is_node<decltype(heap)>) {
        stats.update(outcome);
      } else {
        if (!turn_limit(battle)) {
          stats.update(outcome);
        } else {
          stats.update(outcome);
          // return {0.5, 0.5};
        }
      }

      if (depth == 0) {
        ++output.visit_matrix[outcome.p1.index][outcome.p2.index];
        output.value_matrix[outcome.p1.index][outcome.p2.index] += value.first;
      }

      return value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        float value;
        if constexpr (is_network<decltype(model)>) {
          const auto m = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
              p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
          const auto n = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
              p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
          if (!stats.is_init()) {
            stats.init(m, n);
          }
          if constexpr (is_contextual_bandit<decltype(stats)>) {
            static thread_local std::array<float, 9> p1_logits;
            static thread_local std::array<float, 9> p2_logits;
            value = model.inference(battle, durations(), m, n,
                                    p1_choices.data(), p2_choices.data(),
                                    p1_logits.data(), p2_logits.data());
            stats.softmax_logits(bandit_params, p1_logits.data(),
                                 p2_logits.data());
          } else {
            value = model.inference(battle, durations());
          }
        } else if constexpr (is_poke_engine<decltype(model)>) {
          const auto m = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
              p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
          const auto n = pkmn_gen1_battle_choices(
              &battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
              p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
          if (!stats.is_init()) {
            stats.init(m, n);
          }
          value = model.evaluate(battle);
        } else {
          // model is monte-carlo
          value = init_stats_and_rollout(stats, device, battle, result);
        }
        return {value, 1 - value};
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

  float init_stats_and_rollout(auto &stats, auto &device,
                               pkmn_gen1_battle &battle, pkmn_result result) {

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
    if (!stats.is_init()) {
      stats.init(m, n);
    }
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
      return 1;
    }
    case PKMN_RESULT_LOSE: {
      return 0;
    }
    case PKMN_RESULT_TIE: {
      return 0.5;
    }
    default: {
      assert(false);
      return 0.5;
    }
    };
  }

  inline auto solve_root_matrix_and_sample(auto &device, auto &params,
                                           auto &copy,
                                           const auto &output) noexcept {
    uint8_t p1_index{}, p2_index{};
    const bool periodic_solve = ((output.iterations % params.interval) == 0);
    if (periodic_solve || !initial_solve) {
      // get ucb matrices
      std::array<int, 9 * 9> p1_ucb_matrix;
      std::array<int, 9 * 9> p2_ucb_matrix;
      constexpr int discretize_factor = 256;
      const float log_T = std::log(output.iterations);
      for (auto i = 0; i < output.m; ++i) {
        for (auto j = 0; j < output.n; ++j) {
          float p1_entry = 0;
          float p2_entry = 1;
          if (output.visit_matrix[i][j] < params.minimum) {
            return std::pair<uint8_t, uint8_t>{i, j};
          }
          if (output.visit_matrix[i][j] > 0) {
            p1_entry = output.value_matrix[i][j] / output.visit_matrix[i][j];
            p2_entry = p1_entry;
          }
          const float exploration =
              params.c * std::sqrt(2 * (2 * log_T + ucb_weight) /
                                   (output.visit_matrix[i][j] + 1));
          p1_entry += exploration;
          p2_entry -= exploration;
          p1_ucb_matrix[i * output.n + j] = p1_entry * discretize_factor;
          p2_ucb_matrix[i * output.n + j] = p2_entry * discretize_factor;
        }
      }

      // solve and sample
      std::array<float, 9 + 2> dummy;
      LRSNash::FastInput p1_solve_input{
          static_cast<int>(output.m), static_cast<int>(output.n),
          p1_ucb_matrix.data(), discretize_factor};
      LRSNash::FloatOneSumOutput p1_solve_output{p1_nash.data(), dummy.data(),
                                                 0};
      LRSNash::solve_fast(&p1_solve_input, &p1_solve_output);
      LRSNash::FastInput p2_solve_input{
          static_cast<int>(output.m), static_cast<int>(output.n),
          p2_ucb_matrix.data(), discretize_factor};
      LRSNash::FloatOneSumOutput p2_solve_output{dummy.data(), p2_nash.data(),
                                                 0};
      LRSNash::solve_fast(&p2_solve_input, &p2_solve_output);

      initial_solve = true;
    }

    float p = device.uniform();
    for (auto i = 0; i < output.m; ++i) {
      p -= p1_nash[i];
      if (p <= 0) {
        p1_index = i;
        break;
      }
    }
    p = device.uniform();
    for (auto i = 0; i < output.n; ++i) {
      p -= p2_nash[i];
      if (p <= 0) {
        p2_index = i;
        break;
      }
    }

    return std::pair<uint8_t, uint8_t>{p1_index, p2_index};
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

  // use battle seed to quickly compute a clamped damage roll
  template <size_t n_rolls>
  inline static constexpr uint8_t roll_byte(const uint8_t seed) {
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

  void set_turn_limit(auto &battle) const {
    reinterpret_cast<uint16_t *>(battle.bytes +
                                 PKMN::Layout::Offsets::Battle::turn)[0] = 1000;
  }

  inline bool turn_limit(const auto &battle) const noexcept {
    return reinterpret_cast<const uint16_t *>(
               battle.bytes + PKMN::Layout::Offsets::Battle::turn)[0] >= 1000;
  }

  const auto &durations() const noexcept {
    return *pkmn_gen1_battle_options_chance_durations(&options);
  }

  void process_output(Output &output) {

    // prepare output, solve empirical root matrix if enabled
    // output.empirical_value = output.total_value / output.iterations;
    double total_value = 0;

    // emprically shown to be reliable for 9x9 matrices, 128 bit lrslib
    // TODO maybe use gmp since this is not the hot part of the code

    output.p1_empirical = {};
    output.p2_empirical = {};

    constexpr int discretize_factor = 256;
    std::array<int, 9 * 9> solve_matrix;
    for (int i = 0; i < output.m; ++i) {
      for (int j = 0; j < output.n; ++j) {
        total_value += output.value_matrix[i][j];
        auto n = output.visit_matrix[i][j];
        output.p1_empirical[i] += n;
        output.p2_empirical[j] += n;
        n += !n;
        solve_matrix[output.n * i + j] =
            output.value_matrix[i][j] / n * discretize_factor;
      }
    }

    output.empirical_value = total_value / output.iterations;
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
};

void print_output(const Output &output, const pkmn_gen1_battle &battle,
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

std::string output_string(const MCTS::Output &output,
                          const MCTS::Input &input) {

  std::stringstream ss{};

  constexpr auto label_width = 8;
  const auto &battle = input.battle;
  const auto [p1_labels, p2_labels] = PKMN::choice_labels(battle, input.result);

  auto print_arr = [&ss](const auto &arr, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      ss << std::left << std::fixed << std::setw(label_width)
         << std::setprecision(3) << arr[i] << "  ";
    }
    ss << '\n';
  };

  const auto fix_label = [label_width](auto label) {
    std::stringstream ss{};
    ss << std::left << std::setw(label_width)
       << label.substr(0, label_width - 1);
    return ss.str();
  };

  ss << "Iterations: " << output.iterations
     << ", Time: " << output.duration.count() / 1000.0 << " sec\n";
  ss << "Value: " << std::fixed << std::setprecision(3)
     << output.empirical_value << "\n";

  ss << "\nP1" << std::endl;
  print_arr(p1_labels, output.m);
  print_arr(output.p1_empirical, output.m);
  print_arr(output.p1_nash, output.m);
  ss << "P2" << std::endl;
  print_arr(p2_labels, output.n);
  print_arr(output.p2_empirical, output.n);
  print_arr(output.p2_nash, output.n);

  ss << "\nMatrix:\n";
  std::array<char, label_width + 1> col_offset{};
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.n; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.m; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.n; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        double avg = output.value_matrix[i][j] / output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }

  ss << "\nVisits:\n";
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.n; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.m; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.n; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        auto avg = output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }
  return ss.str();
}

} // namespace MCTS
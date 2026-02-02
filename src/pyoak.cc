#include <encode/battle/battle.h>
#include <encode/battle/frame.h>
#include <encode/battle/policy.h>
#include <encode/build/trajectory.h>
#include <nn/battle/network.h>
#include <nn/default-hyperparameters.h>
#include <train/battle/compressed-frame.h>
#include <train/battle/frame.h>
#include <train/build/trajectory.h>
#include <util/parse.h>
#include <util/search.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

template <std::size_t N, std::size_t M>
std::vector<std::string>
dim_labels_to_vec(const std::array<std::array<char, M>, N> &data) {
  std::vector<std::string> result;
  result.reserve(N);
  for (auto &arr : data) {
    result.emplace_back(arr.data());
  }
  return result;
}

using Tensorizer = Encode::Build::Tensorizer<>;

namespace py = pybind11;

PYBIND11_MODULE(pyoak, m) {
  m.doc() = "Python bindings for oak";

  py::class_<RuntimeSearch::Nodes>(m, "Nodes")
      .def(py::init<>())
      .def("reset", &RuntimeSearch::Nodes::reset);

  py::class_<RuntimeSearch::Agent>(m, "Agent")
      .def(py::init<>())
      .def_readwrite("search_time", &RuntimeSearch::Agent::search_time)
      .def_readwrite("bandit_name", &RuntimeSearch::Agent::bandit_name)
      .def_readwrite("network_path", &RuntimeSearch::Agent::network_path)
      .def_readwrite("matrix_ucb_name", &RuntimeSearch::Agent::matrix_ucb_name)
      .def_readwrite("use_table", &RuntimeSearch::Agent::use_table);
  py::class_<MCTS::Input>(m, "Input").def(py::init<>());

  m.def(
      "parse_battle",
      [](const std::string &battle_string, uint64_t seed = 0x123456) {
        auto [battle, durations] = Parse::parse_battle(battle_string, seed);
        MCTS::Input input{};
        input.battle = battle;
        input.durations = durations;
        input.result = PKMN::result(battle);
        return input;
      },
      py::arg("battle_string"), py::arg("seed") = 0x123456);

  m.def(
      "update",
      [](MCTS::Input &input, uint8_t c1, uint8_t c2) {
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = input.durations;

        pkmn_gen1_battle_options options{};
        pkmn_gen1_battle_options_set(&options, nullptr, &chance_options,
                                     nullptr);

        pkmn_gen1_battle_update(&input.battle, c1, c2, &options);

        input.durations = PKMN::durations(options);
      },
      py::arg("input"), py::arg("c1"), py::arg("c2"));

  m.def(
      "battle_string",
      [](const MCTS::Input &input) {
        return PKMN::battle_data_to_string(input.battle, input.durations);
      },
      py::arg("input"));

  m.def(
      "format",
      [](const MCTS::Input &input, const MCTS::Output &output) {
        return MCTS::output_string(output, input);
      },
      py::arg("input"), py::arg("output"));

  py::class_<MCTS::Output>(m, "Output")
      // allow Python to create one
      .def(py::init<>())

      // scalars
      .def_readonly("m", &MCTS::Output::m)
      .def_readonly("n", &MCTS::Output::n)
      .def_readonly("iterations", &MCTS::Output::iterations)
      .def_readonly("empirical_value", &MCTS::Output::empirical_value)
      .def_readonly("nash_value", &MCTS::Output::nash_value)

      .def_property_readonly(
          "duration_ms",
          [](const MCTS::Output &o) { return o.duration.count(); })

      // 9x9 visit matrix (padded)
      .def_property_readonly("visit_matrix",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<size_t>({9, 9});
                               auto r = arr.mutable_unchecked<2>();
                               for (size_t i = 0; i < 9; ++i)
                                 for (size_t j = 0; j < 9; ++j)
                                   r(i, j) = (i < o.m && j < o.n)
                                                 ? o.visit_matrix[i][j]
                                                 : 0;
                               return arr;
                             })

      // 9x9 value matrix (padded)
      .def_property_readonly("value_matrix",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>({9, 9});
                               auto r = arr.mutable_unchecked<2>();
                               for (size_t i = 0; i < 9; ++i)
                                 for (size_t j = 0; j < 9; ++j)
                                   r(i, j) = (i < o.m && j < o.n)
                                                 ? o.value_matrix[i][j]
                                                 : 0.0;
                               return arr;
                             })

      // 1D vectors
      .def_property_readonly("p1_prior",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1_prior[i];
                               return arr;
                             })

      .def_property_readonly("p2_prior",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p2_prior[i];
                               return arr;
                             })

      .def_property_readonly("p1_empirical",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1_empirical[i];
                               return arr;
                             })

      .def_property_readonly("p2_empirical",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p2_empirical[i];
                               return arr;
                             })

      .def_property_readonly("p1_nash",
                             [](const MCTS::Output &o) {
                               auto arr = py::array_t<double>(9);
                               auto r = arr.mutable_unchecked<1>();
                               for (size_t i = 0; i < 9; ++i)
                                 r(i) = o.p1_nash[i];
                               return arr;
                             })

      .def_property_readonly("p2_nash", [](const MCTS::Output &o) {
        auto arr = py::array_t<double>(9);
        auto r = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < 9; ++i)
          r(i) = o.p2_nash[i];
        return arr;
      });
  m.def(
      "search",
      [](const MCTS::Input &input, RuntimeSearch::Nodes &nodes,
         RuntimeSearch::Agent &agent, MCTS::Output output = {}) {
        mt19937 device{std::random_device{}()};
        return RuntimeSearch::run(device, input, nodes, agent, output);
      },
      py::arg("input"), py::arg("nodes"), py::arg("agent"),
      py::arg("output") = MCTS::Output{});

  // Battle net hyperparams
  m.attr("pokemon_in_dim") = Encode::Battle::Pokemon::n_dim;
  m.attr("active_in_dim") = Encode::Battle::Active::n_dim;
  m.attr("pokemon_hidden_dim") = NN::Battle::Default::pokemon_hidden_dim;
  m.attr("pokemon_out_dim") = NN::Battle::Default::pokemon_out_dim;
  m.attr("active_hidden_dim") = NN::Battle::Default::active_hidden_dim;
  m.attr("active_out_dim") = NN::Battle::Default::active_out_dim;
  m.attr("side_out_dim") = NN::Battle::Default::side_out_dim;
  m.attr("hidden_dim") = NN::Battle::Default::hidden_dim;
  m.attr("value_hidden_dim") = NN::Battle::Default::value_hidden_dim;
  m.attr("policy_hidden_dim") = NN::Battle::Default::policy_hidden_dim;
  m.attr("policy_out_dim") = NN::Battle::Default::policy_out_dim;

  // Build net hyperparams
  // m.attr("build_policy_hidden_dim") =
  //     NN::Build::Default::build_policy_hidden_dim;
  // m.attr("build_value_hidden_dim") =
  // NN::Build::Default::build_value_hidden_dim; m.attr("build_max_actions") =
  // NN::Build::Default::build_max_actions;

  m.def("move_names",
        []() { return dim_labels_to_vec(PKMN::Data::MOVE_CHAR_ARRAY); });

  m.def("species_names",
        []() { return dim_labels_to_vec(PKMN::Data::SPECIES_CHAR_ARRAY); });

  m.def("pokemon_dim_labels", []() {
    return dim_labels_to_vec(Encode::Battle::Pokemon::dim_labels);
  });

  m.def("active_dim_labels",
        []() { return dim_labels_to_vec(Encode::Battle::Active::dim_labels); });

  m.def("policy_dim_labels", []() {
    auto v = dim_labels_to_vec(Encode::Battle::Policy::dim_labels);
    v.push_back(""); // preserve your extra empty string
    return v;
  });

  // Species-move list
  m.def("species_move_list", []() {
    std::vector<std::pair<int, int>> result;
    result.reserve(Tensorizer::species_move_list_size);
    for (int i = 0; i < Tensorizer::species_move_list_size; ++i) {
      auto p = Tensorizer::species_move_list(i);
      result.emplace_back(static_cast<int>(p.first),
                          static_cast<int>(p.second));
    }
    return result;
  });
}

extern "C" int index_compressed_battle_frames(const char *path, char *out_data,
                                              uint16_t *offsets,
                                              uint16_t *frame_counts) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  int n_battles = 0;
  size_t total_offset = 0;
  while (true) {

    if (file.peek() == EOF) {
      break;
    }

    uint32_t offset;
    uint16_t frame_count;
    file.read(reinterpret_cast<char *>(&offset), 4);
    if (file.gcount() < 4) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.read(reinterpret_cast<char *>(&frame_count), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad frame count read" << std::endl;
      return -1;
    }
    file.seekg(-6, std::ios::cur);

    std::vector<char> buffer;
    buffer.reserve(offset);
    file.read(buffer.data(), offset);

    if (file.gcount() < offset) {
      std::cerr << "truncated battle frame" << std::endl;
      return -1;
    }

    std::memcpy(out_data + total_offset, buffer.data(), offset);
    offsets[n_battles] = offset;
    frame_counts[n_battles] = frame_count;
    total_offset += offset;
    ++n_battles;
  }

  return n_battles;
}

extern "C" void uncompress_training_frames(
    const char *data, uint8_t *m, uint8_t *n, uint8_t *battle,
    uint8_t *durations, uint8_t *result, uint8_t *p1_choices,
    uint8_t *p2_choices, uint32_t *iterations, float *p1_empirical,
    float *p1_nash, float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  Train::Battle::CompressedFrames compressed_frames{};
  compressed_frames.read(data);

  using In = Train::Battle::FrameInput;
  In input{.m = m,
           .n = n,
           .battle = battle,
           .durations = durations,
           .result = result,
           .p1_choices = p1_choices,
           .p2_choices = p2_choices,
           .iterations = iterations,
           .p1_empirical = p1_empirical,
           .p1_nash = p1_nash,
           .p2_empirical = p2_empirical,
           .p2_nash = p2_nash,
           .empirical_value = empirical_value,
           .nash_value = nash_value,
           .score = score};

  const auto frames = compressed_frames.uncompress();
  for (const auto frame : frames) {
    input.write(frame);
  }
}

extern "C" void uncompress_and_encode_training_frames(
    const char *data, uint8_t *m, uint8_t *n, int64_t *p1_choice_indices,
    int64_t *p2_choice_indices, float *pokemon, float *active, float *hp,
    uint32_t *iterations, float *p1_empirical, float *p1_nash,
    float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  Train::Battle::CompressedFrames compressed_frames{};
  compressed_frames.read(data);

  Encode::Battle::FrameInput input{.m = m,
                                   .n = n,
                                   .p1_choice_indices = p1_choice_indices,
                                   .p2_choice_indices = p2_choice_indices,
                                   .pokemon = pokemon,
                                   .active = active,
                                   .hp = hp,
                                   .iterations = iterations,
                                   .p1_empirical = p1_empirical,
                                   .p1_nash = p1_nash,
                                   .p2_empirical = p2_empirical,
                                   .p2_nash = p2_nash,
                                   .empirical_value = empirical_value,
                                   .nash_value = nash_value,
                                   .score = score};

  const auto frames = compressed_frames.uncompress();
  for (const auto frame : frames) {
    Encode::Battle::Frame encoded{frame};
    input.write(encoded, frame.target);
  }
}

extern "C" int test_consistency(size_t max_games, const char *network_path,
                                const char *path, char *out_data,
                                uint16_t *offsets, uint16_t *frame_counts) {

  std::ifstream network_params{network_path, std::ios::binary};
  NN::Battle::Network network{};
  if (!network.read_parameters(network_params)) {
    return -1;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  int n_battles = 0;
  size_t total_offset = 0;
  for (auto i = 0; i < max_games; ++i) {

    if (file.peek() == EOF) {
      break;
    }

    uint32_t offset;
    uint16_t frame_count;
    file.read(reinterpret_cast<char *>(&offset), 4);
    if (file.gcount() < 4) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.read(reinterpret_cast<char *>(&frame_count), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad frame count read" << std::endl;
      return -1;
    }
    file.seekg(-6, std::ios::cur);

    std::vector<char> buffer;
    buffer.reserve(offset);
    file.read(buffer.data(), offset);

    if (file.gcount() < offset) {
      std::cerr << "truncated battle frame" << std::endl;
      return -1;
    }

    Train::Battle::CompressedFrames compressed_frames{};
    compressed_frames.read(buffer.data());

    auto battle = compressed_frames.battle;
    auto options = PKMN::options();
    auto result = PKMN::result();

    network.fill_pokemon_caches(battle);

    std::vector<float> network_values{};
    network_values.reserve(compressed_frames.updates.size());

    std::array<pkmn_choice, 9> p1_choices;
    std::array<pkmn_choice, 9> p2_choices;

    for (const auto &update : compressed_frames.updates) {
      const auto m = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), p1_choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      const auto n = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), p2_choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      std::array<float, 9> p1_logits{};
      std::array<float, 9> p2_logits{};

      const auto durations = PKMN::durations(options);
      const auto value = network.inference(battle, durations, m, n,
                                           p1_choices.data(), p2_choices.data(),
                                           p1_logits.data(), p2_logits.data());
      network_values.push_back(value);
      // std::cout << "____" << std::endl;
      std::cout << value << ' ';
      // for (auto i = 0; i < m; ++i) {
      //   std::cout << p1_logits[i] << ' ';
      // }
      // std::cout << std::endl;
      // for (auto i = 0; i < n; ++i) {
      //   std::cout << p2_logits[i] << ' ';
      // }
      // std::cout << std::endl;

      result = PKMN::update(battle, update.c1, update.c2, options);
    }

    std::cout << compressed_frames.updates.size() << std::endl;

    std::memcpy(out_data + total_offset, buffer.data(), offset);
    offsets[n_battles] = offset;
    frame_counts[n_battles] = frame_count;
    total_offset += offset;
    ++n_battles;

    // break;
  }

  return n_battles;
}

extern "C" size_t sample_from_battle_data_files(
    size_t max_count, size_t threads, size_t max_battle_length,
    size_t min_interations, // params
    size_t n_paths, const char *const *paths, const int *n_battles,
    const int *const *offsets,
    const int *const *n_frames, // input
    uint8_t *m, uint8_t *n, int64_t *p1_choice_indices,
    int64_t *p2_choice_indices, float *pokemon, float *active, float *hp,
    uint32_t *iterations, float *p1_empirical, float *p1_nash,
    float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score

) {
  using Input = Encode::Battle::FrameInput;
  Input input{m,
              n,
              p1_choice_indices,
              p2_choice_indices,
              pokemon,
              active,
              hp,
              iterations,
              p1_empirical,
              p1_nash,
              p2_empirical,
              p2_nash,
              empirical_value,
              nash_value,
              score};

  const auto ptrs = std::bit_cast<std::array<void *, 15>>(input);
  if (std::any_of(ptrs.begin(), ptrs.end(), [](const auto x) { return !x; })) {
    std::cerr << "null pointer in input" << std::endl;
    return 0;
  }

  size_t total_battles = 0;
  for (auto i = 0; i < n_paths; ++i) {
    total_battles += n_battles[i];
  }

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() -> void {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist_int{0, total_battles - 1};
    std::vector<char> buffer{};

    const auto report_error = [&](const auto &msg) -> void {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
      return;
    };

    try {

      while (!errors.load()) {

        const auto get_indices = [&]() -> std::pair<uint32_t, uint32_t> {
          auto bi = dist_int(mt);
          auto pi = 0;
          while (bi >= 0) {
            const auto n_battles_in_path = n_battles[pi];
            if (n_battles_in_path > bi) {
              break;
            } else {
              bi -= n_battles_in_path;
              ++pi;
            }
          }
          return {bi, pi};
        };

        const auto [battle_index, path_index] = get_indices();

        if (path_index >= n_paths) {
          return report_error("bad path index");
        }

        const char *path = paths[path_index];
        std::ifstream file(path, std::ios::binary);
        if (!file) {
          return report_error("unable to open file");
        }

        const auto battle_offset = offsets[path_index][battle_index];

        file.seekg(battle_offset, std::ios::beg);

        using Offset = Train::Battle::CompressedFrames::Offset;
        using FrameCount = Train::Battle::CompressedFrames::FrameCount;

        Offset offset;
        FrameCount n_frames;

        file.read(reinterpret_cast<char *>(&offset), sizeof(Offset));
        if (file.gcount() < sizeof(Offset)) {
          return report_error("bad offset read");
        }
        file.read(reinterpret_cast<char *>(&n_frames), sizeof(FrameCount));
        if (file.gcount() < sizeof(FrameCount)) {
          return report_error("bad frame count read");
        }

        if (n_frames > max_battle_length) {
          continue;
        }

        file.seekg(-(sizeof(Offset) + sizeof(FrameCount)), std::ios::cur);
        if ((offset > 200000) || (offset < sizeof(pkmn_gen1_battle))) {
          return report_error("Bad offset length: " + std::to_string(offset) +
                              "; frames: " + std::to_string(n_frames));
        }
        buffer.resize(offset);
        buffer.clear();
        file.read(buffer.data(), offset);
        Train::Battle::CompressedFrames compressed_frames{};
        compressed_frames.read(buffer.data());

        std::vector<int> valid_frame_indices{};
        for (auto i = 0; i < compressed_frames.updates.size(); ++i) {
          const auto &update = compressed_frames.updates[i];
          if (update.iterations >= min_interations) {
            valid_frame_indices.push_back(i);
          }
        }

        if (valid_frame_indices.size() == 0) {
          continue;
        }

        const auto selected_frame_index =
            valid_frame_indices[std::uniform_int_distribution<size_t>{
                0, valid_frame_indices.size() - 1}(mt)];

        const auto frames = compressed_frames.uncompress();
        const auto &frame = frames[selected_frame_index];

        const auto write_index = count.fetch_add(1);
        if (write_index >= max_count) {
          return;
        } else {
          input.index(write_index)
              .write(Encode::Battle::Frame{frame}, frame.target);
        }
      }

    } catch (const std::exception &e) {
      report_error(e.what());
    }
  };

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return errors.load() ? 0 : std::min(count.load(), max_count);
}

extern "C" size_t
read_build_trajectories(size_t max_count, size_t threads, size_t n_paths,
                        const char *const *paths, int64_t *action,
                        int64_t *mask, float *policy, float *value,
                        float *score, int64_t *start, int64_t *end) {

  constexpr auto traj_size =
      Encode::Build::CompressedTrajectory<>::size_no_team;

  using In = Encode::Build::TrajectoryInput<>;
  In input{.action = action,
           .mask = mask,
           .policy = policy,
           .value = value,
           .score = score,
           .start = start,
           .end = end};

  const auto ptrs = std::bit_cast<std::array<void *, 7>>(input);
  if (std::any_of(ptrs.begin(), ptrs.end(), [](const auto x) { return !x; })) {
    std::cerr << "null pointer in input" << std::endl;
    return 0;
  }

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> file_dist{0, n_paths - 1};

    const auto report_error = [&](const auto &msg) -> void {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
      return;
    };

    try {
      while (true) {

        const auto path_index = file_dist(mt);
        std::ifstream file(paths[path_index], std::ios::binary);
        if (!file) {
          return report_error("Failed to open file " +
                              std::to_string(path_index));
        }
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0);
        if ((size % traj_size) != 0) {
          return report_error("File " + std::to_string(path_index) + " size " +
                              std::to_string(size) + " is not a multiple of " +
                              std::to_string(traj_size));
        }

        const auto n_trajectories = size / traj_size;

        const auto trajectory_index =
            std::uniform_int_distribution<size_t>{0, n_trajectories - 1}(mt);
        file.seekg(trajectory_index * traj_size);

        Encode::Build::CompressedTrajectory<> traj;
        file.read(reinterpret_cast<char *>(&traj), traj_size);
        if (file.gcount() < traj_size) {
          return report_error("Bad trajectory read");
        }
        const auto format = static_cast<uint8_t>(traj.header.format);
        if (format != 0) {
          return report_error("Only NoTeam trajectories are supported");
        }

        const auto write_index = count.fetch_add(1);
        if (write_index >= max_count) {
          return;
        } else {
          input.index(write_index).write(traj);
        }
      }
    } catch (const std::exception &e) {
      return report_error(e.what());
    }
  };

  const auto start_ = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  const auto end_ = std::chrono::high_resolution_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
  // std::cout << ms.count() << std::endl;
  return errors.load() ? 0 : std::min(count.load(), max_count);
}

extern "C" void print_battle_data(uint8_t *battle_bytes,
                                  uint8_t *durations_bytes) {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  std::copy(battle_bytes, battle_bytes + PKMN::Layout::Sizes::Battle,
            battle.bytes);
  std::copy(durations_bytes, durations_bytes + PKMN::Layout::Sizes::Durations,
            durations.bytes);
  std::cout << PKMN::battle_data_to_string(battle, durations) << std::endl;
}
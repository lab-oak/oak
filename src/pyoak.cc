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

struct BattleFrame {
  size_t size;

  py::array_t<uint8_t> m;
  py::array_t<uint8_t> n;
  py::array_t<uint8_t> battle;
  py::array_t<uint8_t> durations;
  py::array_t<uint8_t> result;

  py::array_t<uint8_t> p1_choices;
  py::array_t<uint8_t> p2_choices;

  py::array_t<uint32_t> iterations;
  py::array_t<float> p1_empirical;
  py::array_t<float> p1_nash;
  py::array_t<float> p2_empirical;
  py::array_t<float> p2_nash;

  py::array_t<float> empirical_value;
  py::array_t<float> nash_value;
  py::array_t<float> score;

  BattleFrame(size_t size_) : size(size_) {
    std::vector<ssize_t> shape1{static_cast<ssize_t>(size), 1};
    std::vector<ssize_t> shape8{static_cast<ssize_t>(size), 8};
    std::vector<ssize_t> shape9{static_cast<ssize_t>(size), 9};
    std::vector<ssize_t> shape384{static_cast<ssize_t>(size), 384};

    m = py::array_t<uint8_t>(shape1);
    n = py::array_t<uint8_t>(shape1);
    battle = py::array_t<uint8_t>(shape384);
    durations = py::array_t<uint8_t>(shape8);
    result = py::array_t<uint8_t>(shape1);
    p1_choices = py::array_t<uint8_t>(shape9);
    p2_choices = py::array_t<uint8_t>(shape9);
    iterations = py::array_t<uint32_t>(shape1);
    p1_empirical = py::array_t<float>(shape9);
    p1_nash = py::array_t<float>(shape9);
    p2_empirical = py::array_t<float>(shape9);
    p2_nash = py::array_t<float>(shape9);
    empirical_value = py::array_t<float>(shape1);
    nash_value = py::array_t<float>(shape1);
    score = py::array_t<float>(shape1);
  }

  void uncompress_from_bytes(const py::bytes &data) {
    std::string_view data_view(data);
    const char *raw_data = data_view.data();

    Train::Battle::CompressedFrames compressed_frames{};
    compressed_frames.read(raw_data);

    using In = Train::Battle::FrameInput;
    In input{.m = m.mutable_data(),
             .n = n.mutable_data(),
             .battle = battle.mutable_data(),
             .durations = durations.mutable_data(),
             .result = result.mutable_data(),
             .p1_choices = p1_choices.mutable_data(),
             .p2_choices = p2_choices.mutable_data(),
             .iterations = iterations.mutable_data(),
             .p1_empirical = p1_empirical.mutable_data(),
             .p1_nash = p1_nash.mutable_data(),
             .p2_empirical = p2_empirical.mutable_data(),
             .p2_nash = p2_nash.mutable_data(),
             .empirical_value = empirical_value.mutable_data(),
             .nash_value = nash_value.mutable_data(),
             .score = score.mutable_data()};

    const auto frames_vec = compressed_frames.uncompress();
    for (const auto &frame : frames_vec) {
      input.write(frame);
    }
  }

  // optional: factory function
  static BattleFrame from_bytes(const py::bytes &data, size_t size) {
    BattleFrame f(size);
    f.uncompress_from_bytes(data);
    return f;
  }
};

struct EncodedBattleFrame {
  size_t size;
  py::array_t<uint8_t> m;
  py::array_t<uint8_t> n;
  py::array_t<int64_t> p1_choice_indices;
  py::array_t<int64_t> p2_choice_indices;
  py::array_t<float> pokemon; // shape: (size, 2, 5, pokemon_in_dim)
  py::array_t<float> active;  // shape: (size, 2, 1, active_in_dim)
  py::array_t<float> hp;      // shape: (size, 2, 6, 1)
  py::array_t<uint32_t> iterations;
  py::array_t<float> p1_empirical;
  py::array_t<float> p1_nash;
  py::array_t<float> p2_empirical;
  py::array_t<float> p2_nash;
  py::array_t<float> empirical_value;
  py::array_t<float> nash_value;
  py::array_t<float> score;

  static constexpr ssize_t pokemon_in_dim = Encode::Battle::Pokemon::n_dim;
  static constexpr ssize_t active_in_dim = Encode::Battle::Active::n_dim;

  EncodedBattleFrame(size_t sz) : size(sz) {
    auto make_shape = [sz](std::vector<ssize_t> dims) {
      dims[0] = static_cast<ssize_t>(sz); // overwrite first dim with batch size
      return dims;
    };
    m = py::array_t<uint8_t>(make_shape({0, 1}));
    n = py::array_t<uint8_t>(make_shape({0, 1}));
    p1_choice_indices = py::array_t<int64_t>(make_shape({0, 9}));
    p2_choice_indices = py::array_t<int64_t>(make_shape({0, 9}));
    pokemon = py::array_t<float>(make_shape({0, 2, 5, pokemon_in_dim}));
    active = py::array_t<float>(make_shape({0, 2, 1, active_in_dim}));
    hp = py::array_t<float>(make_shape({0, 2, 6, 1}));
    iterations = py::array_t<uint32_t>(make_shape({0, 1}));
    p1_empirical = py::array_t<float>(make_shape({0, 9}));
    p1_nash = py::array_t<float>(make_shape({0, 9}));
    p2_empirical = py::array_t<float>(make_shape({0, 9}));
    p2_nash = py::array_t<float>(make_shape({0, 9}));
    empirical_value = py::array_t<float>(make_shape({0, 1}));
    nash_value = py::array_t<float>(make_shape({0, 1}));
    score = py::array_t<float>(make_shape({0, 1}));
  }

  void clear() {
    std::fill_n(m.mutable_data(), m.size(), uint8_t(0));
    std::fill_n(n.mutable_data(), n.size(), uint8_t(0));

    std::fill_n(p1_choice_indices.mutable_data(), p1_choice_indices.size(),
                int64_t(0));
    std::fill_n(p2_choice_indices.mutable_data(), p2_choice_indices.size(),
                int64_t(0));

    std::fill_n(pokemon.mutable_data(), pokemon.size(), 0.0f);
    std::fill_n(active.mutable_data(), active.size(), 0.0f);
    std::fill_n(hp.mutable_data(), hp.size(), 0.0f);

    std::fill_n(iterations.mutable_data(), iterations.size(), uint32_t(0));
    std::fill_n(p1_empirical.mutable_data(), p1_empirical.size(), 0.0f);
    std::fill_n(p1_nash.mutable_data(), p1_nash.size(), 0.0f);
    std::fill_n(p2_empirical.mutable_data(), p2_empirical.size(), 0.0f);
    std::fill_n(p2_nash.mutable_data(), p2_nash.size(), 0.0f);

    std::fill_n(empirical_value.mutable_data(), empirical_value.size(), 0.0f);
    std::fill_n(nash_value.mutable_data(), nash_value.size(), 0.0f);
    std::fill_n(score.mutable_data(), score.size(), 0.0f);
  }

  // member function to uncompress and encode
  void uncompress_from_bytes(const py::bytes &data) {
    std::string_view sv(data);
    const char *raw_data = sv.data();

    Train::Battle::CompressedFrames compressed_frames{};
    compressed_frames.read(raw_data);

    Encode::Battle::FrameInput input{
        .m = m.mutable_data(),
        .n = n.mutable_data(),
        .p1_choice_indices = p1_choice_indices.mutable_data(),
        .p2_choice_indices = p2_choice_indices.mutable_data(),
        .pokemon = pokemon.mutable_data(),
        .active = active.mutable_data(),
        .hp = hp.mutable_data(),
        .iterations = iterations.mutable_data(),
        .p1_empirical = p1_empirical.mutable_data(),
        .p1_nash = p1_nash.mutable_data(),
        .p2_empirical = p2_empirical.mutable_data(),
        .p2_nash = p2_nash.mutable_data(),
        .empirical_value = empirical_value.mutable_data(),
        .nash_value = nash_value.mutable_data(),
        .score = score.mutable_data()};

    const auto frames_vec = compressed_frames.uncompress();
    for (const auto &frame : frames_vec) {
      Encode::Battle::Frame encoded{frame};
      input.write(encoded, frame.target);
    }
  }

  // optional static factory
  static EncodedBattleFrame from_bytes(const py::bytes &data, size_t sz) {
    EncodedBattleFrame f(sz);
    f.uncompress_from_bytes(data);
    return f;
  }
};

py::list read_battle_data(const std::string &path,
                          size_t max_battles = 1'000'000) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  py::list result;
  size_t total_battles = 0;

  while (file.peek() != EOF && total_battles < max_battles) {
    uint32_t offset;
    uint16_t frame_count;

    file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    if (file.gcount() != sizeof(offset))
      throw std::runtime_error("bad offset read");

    file.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
    if (file.gcount() != sizeof(frame_count))
      throw std::runtime_error("bad frame count read");

    // move back 6 bytes to match original seek
    file.seekg(-6, std::ios::cur);

    std::vector<char> buffer(offset);
    file.read(buffer.data(), offset);
    if (file.gcount() != offset)
      throw std::runtime_error("truncated battle frame");

    result.append(py::make_tuple(py::bytes(buffer.data(), buffer.size()),
                                 static_cast<int>(frame_count)));

    ++total_battles;
  }

  return result;
}

struct SampleIndexer {
  // store Python objects directly
  std::unordered_map<std::string, py::list> data;

  SampleIndexer() = default;

  size_t size() const { return data.size(); }

  py::list get(const std::string &path) {
    auto it = data.find(path);
    if (it != data.end()) {
      return it->second; // cached Python list, no conversion needed
    }

    py::list output;
    int total_offset = 0;

    py::object py_battle_data =
        read_battle_data(path); // Python list of (bytes,int)

    for (auto bf : py_battle_data) {
      py::tuple t = bf.cast<py::tuple>();
      py::bytes battle_bytes = t[0].cast<py::bytes>();
      int frame_count = t[1].cast<int>();

      output.append(py::make_tuple(total_offset, frame_count));
      total_offset += static_cast<int>(PyBytes_Size(battle_bytes.ptr()));
    }

    data[path] = output; // now works because data is py::list
    return output;
  }

  void prune(const std::vector<std::string> &paths) {
    for (auto it = data.begin(); it != data.end();) {
      if (std::find(paths.begin(), paths.end(), it->first) == paths.end()) {
        it = data.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Pybind version of sample_from_battle_data_files
  size_t sample(EncodedBattleFrame &encoded_frames, size_t threads,
                size_t max_game_length = 10000, size_t minimum_iterations = 1) {
    size_t max_count = encoded_frames.size;

    // prepare C++ structures from cached Python data
    std::vector<const char *> paths;
    std::vector<int> n_battles;
    std::vector<std::vector<int>> offsets;
    std::vector<std::vector<int>> n_frames;

    for (auto &p : data) {
      std::string path_str = p.first;
      paths.push_back(path_str.c_str());

      py::list lst = p.second;
      n_battles.push_back(py::len(lst));

      offsets.emplace_back();
      n_frames.emplace_back();

      for (auto item : lst) {
        py::tuple t = item.cast<py::tuple>();
        offsets.back().push_back(t[0].cast<int>());
        n_frames.back().push_back(t[1].cast<int>());
      }
    }

    std::cout << data.size() << std::endl;

    // convert vectors to pointers for the C++ sampler
    std::vector<const int *> offsets_pp, n_frames_pp;
    for (size_t i = 0; i < offsets.size(); i++) {
      offsets_pp.push_back(offsets[i].data());
      n_frames_pp.push_back(n_frames[i].data());
    }

    std::atomic<size_t> count{};
    std::atomic<size_t> errors{};

    auto start_reading = [&]() {
      std::mt19937 mt{std::random_device{}()};
      std::uniform_int_distribution<size_t> dist_int{0,
                                                     encoded_frames.size - 1};

      while (!errors.load()) {
        // Pick a random battle
        size_t total_battles = 0;
        for (auto n : n_battles)
          total_battles += n;
        if (total_battles == 0)
          return;

        size_t bi = dist_int(mt);
        size_t path_index = 0;
        while (bi >= n_battles[path_index]) {
          bi -= n_battles[path_index];
          path_index++;
        }
        size_t battle_index = bi;

        // TODO: open file, read battle, select valid frame
        // This part can directly call your existing C++ sampler code
        // using encoded_frames mutable_data() as destination
      }
    };

    // launch threads
    std::vector<std::thread> pool;
    for (size_t i = 0; i < threads; i++)
      pool.emplace_back(start_reading);
    for (auto &t : pool)
      t.join();

    return errors.load() ? 0 : std::min(count.load(), max_count);
  }
};

namespace py = pybind11;

PYBIND11_MODULE(py_oak, m) {
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
      .def(py::init<>())
      .def_readonly("m", &MCTS::Output::m)
      .def_readonly("n", &MCTS::Output::n)
      .def_readonly("iterations", &MCTS::Output::iterations)
      .def_readonly("empirical_value", &MCTS::Output::empirical_value)
      .def_readonly("nash_value", &MCTS::Output::nash_value)

      .def_property_readonly(
          "duration_ms",
          [](const MCTS::Output &o) { return o.duration.count(); })
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

  py::class_<BattleFrame>(m, "BattleFrame")
      .def(py::init<size_t>())
      .def("uncompress_from_bytes", &BattleFrame::uncompress_from_bytes)
      .def_static("from_bytes", &BattleFrame::from_bytes)
      .def_readonly("size", &BattleFrame::size);

  py::class_<EncodedBattleFrame>(m, "EncodedBattleFrame")
      .def(py::init<size_t>())
      .def("clear", &EncodedBattleFrame::clear)
      .def_readonly("size", &EncodedBattleFrame::size)
      .def_readonly("m", &EncodedBattleFrame::m)
      .def_readonly("n", &EncodedBattleFrame::n)
      .def_readonly("p1_choice_indices", &EncodedBattleFrame::p1_choice_indices)
      .def_readonly("p2_choice_indices", &EncodedBattleFrame::p2_choice_indices)
      .def_readonly("pokemon", &EncodedBattleFrame::pokemon)
      .def_readonly("active", &EncodedBattleFrame::active)
      .def_readonly("hp", &EncodedBattleFrame::hp)
      .def_readonly("iterations", &EncodedBattleFrame::iterations)
      .def_readonly("p1_empirical", &EncodedBattleFrame::p1_empirical)
      .def_readonly("p1_nash", &EncodedBattleFrame::p1_nash)
      .def_readonly("p2_empirical", &EncodedBattleFrame::p2_empirical)
      .def_readonly("p2_nash", &EncodedBattleFrame::p2_nash)
      .def_readonly("empirical_value", &EncodedBattleFrame::empirical_value)
      .def_readonly("nash_value", &EncodedBattleFrame::nash_value)
      .def_readonly("score", &EncodedBattleFrame::score);

  // pybind11 module
  py::class_<SampleIndexer>(m, "SampleIndexer")
      .def(py::init<>())
      .def("get", &SampleIndexer::get)
      .def("prune", &SampleIndexer::prune)
      .def("sample", &SampleIndexer::sample)
      .def("size", &SampleIndexer::size);

  m.def("read_battle_data", &read_battle_data, py::arg("path"),
        py::arg("max_battles") = 1'000'000);
}

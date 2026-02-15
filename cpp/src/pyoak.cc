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

namespace {

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

py::list read_battle_data(const std::string &path,
                          size_t max_battles = 1'000'000) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("read_battle_data: Failed to open file: " + path);
  }
  py::list result;
  size_t total_battles = 0;
  std::vector<char> buffer{};
  while (file.peek() != EOF && total_battles < max_battles) {
    uint32_t offset;
    uint16_t frame_count;
    file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    if (file.gcount() != sizeof(offset))
      throw std::runtime_error("read_battle_data: bad offset read");
    file.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count));
    if (file.gcount() != sizeof(frame_count))
      throw std::runtime_error("read_battle_data: bad frame count read");
    file.seekg(-(sizeof(offset) + sizeof(frame_count)), std::ios::cur);
    buffer.resize(std::max(static_cast<size_t>(offset), buffer.size()));
    file.read(buffer.data(), offset);
    if (file.gcount() != offset) {
      throw std::runtime_error("read_battle_data: bad battle buffer read");
    }
    result.append(py::make_tuple(py::bytes(buffer.data(), offset),
                                 static_cast<int>(frame_count)));
    ++total_battles;
  }
  return result;
}

struct SampleIndexer {
  std::unordered_map<std::string, py::list> data;

  SampleIndexer() = default;

  size_t size() const { return data.size(); }

  py::list get(const std::string &path) {
    auto it = data.find(path);
    if (it != data.end()) {
      return it->second;
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
    data[path] = output;
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
};

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
    std::vector<size_t> shape1{static_cast<size_t>(size), 1};
    std::vector<size_t> shape8{static_cast<size_t>(size), 8};
    std::vector<size_t> shape9{static_cast<size_t>(size), 9};
    std::vector<size_t> shape384{static_cast<size_t>(size), 384};

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
  py::array_t<float> pokemon;
  py::array_t<float> active;
  py::array_t<float> hp;
  py::array_t<uint32_t> iterations;
  py::array_t<float> p1_empirical;
  py::array_t<float> p1_nash;
  py::array_t<float> p2_empirical;
  py::array_t<float> p2_nash;
  py::array_t<float> empirical_value;
  py::array_t<float> nash_value;
  py::array_t<float> score;

  static constexpr size_t pokemon_in_dim = Encode::Battle::Pokemon::n_dim;
  static constexpr size_t active_in_dim = Encode::Battle::Active::n_dim;

  EncodedBattleFrame(size_t sz) : size(sz) {
    auto make_shape = [sz](std::vector<size_t> dims) {
      dims[0] = static_cast<size_t>(sz); // overwrite first dim with batch size
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

  size_t sample(const SampleIndexer &indexer, size_t threads,
                size_t max_battle_length, size_t min_iterations) {
    using Input = Encode::Battle::FrameInput;

    Input input{
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
        .score = score.mutable_data(),
    };

    const auto ptrs = std::bit_cast<std::array<void *, 15>>(input);
    if (std::any_of(ptrs.begin(), ptrs.end(), [](auto p) { return !p; })) {
      std::cerr << "null pointer in input" << std::endl;
      return 0;
    }

    // flatten indexer data into C++ arrays
    std::vector<const char *> paths;
    std::vector<int> n_battles;
    std::vector<std::vector<int>> offsets;
    std::vector<std::vector<int>> n_frames;
    for (const auto &[path, lst] : indexer.data) {
      paths.push_back(path.c_str());
      n_battles.push_back(py::len(lst));
      offsets.emplace_back();
      n_frames.emplace_back();
      for (auto item : lst) {
        py::tuple t = item.cast<py::tuple>();
        offsets.back().push_back(t[0].cast<int>());
        n_frames.back().push_back(t[1].cast<int>());
      }
    }

    size_t total_battles = 0;
    for (auto n : n_battles) {
      total_battles += n;
    }

    std::atomic<size_t> count{};
    std::atomic<size_t> errors{};

    const auto start_reading = [&]() {
      std::mt19937 mt{std::random_device{}()};
      std::uniform_int_distribution<size_t> battle_dist(0, total_battles - 1);
      std::vector<char> buffer{};

      auto report_error = [&](const std::string &msg) {
        std::cerr << msg << std::endl;
        errors.fetch_add(1);
      };

      try {
        while (!errors.load()) {
          // battle_index is sampled globally and then subtracted until the
          // path_index is found and then battle_index is local to the path's
          // data
          size_t battle_index = battle_dist(mt);
          size_t path_index = 0;
          while (battle_index >= static_cast<size_t>(n_battles[path_index])) {
            battle_index -= n_battles[path_index];
            ++path_index;
          }
          if (path_index >= paths.size()) {
            report_error("bad path index");
            return;
          }

          std::ifstream file(paths[path_index], std::ios::binary);
          if (!file) {
            report_error("unable to open file");
            return;
          }

          const auto battle_offset = offsets[path_index][battle_index];
          file.seekg(battle_offset, std::ios::beg);

          auto offset = Train::Battle::CompressedFrames::Offset{};
          auto frame_count = Train::Battle::CompressedFrames::FrameCount{};

          file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
          if (file.gcount() < sizeof(offset)) {
            report_error("bad offset read");
            return;
          }
          file.read(reinterpret_cast<char *>(&frame_count),
                    sizeof(frame_count));
          if (file.gcount() < sizeof(frame_count)) {
            report_error("bad frame count read");
            return;
          }
          if (frame_count > max_battle_length) {
            continue;
          }
          file.seekg(-(sizeof(offset) + sizeof(frame_count)), std::ios::cur);
          if (offset > 200000 || offset < sizeof(pkmn_gen1_battle)) {
            report_error("bad offset length");
            return;
          }

          buffer.resize(std::max(static_cast<size_t>(offset), buffer.size()));
          file.read(buffer.data(), offset);

          Train::Battle::CompressedFrames compressed;
          compressed.read(buffer.data());

          std::vector<size_t> valid;
          for (size_t i = 0; i < compressed.updates.size(); ++i) {
            if (compressed.updates[i].iterations >= min_iterations)
              valid.push_back(i);
          }
          if (valid.empty()) {
            continue;
          }
          const auto frames = compressed.uncompress();
          const auto selected = valid[std::uniform_int_distribution<size_t>(
              0, valid.size() - 1)(mt)];
          const auto &frame = frames[selected];
          size_t write_index = count.fetch_add(1);
          if (write_index >= size) {
            return;
          }
          // encode the frame and write to tensor
          input.index(write_index)
              .write(Encode::Battle::Frame{frame}, frame.target);
        }
      } catch (const std::exception &e) {
        report_error(e.what());
      }
    };

    std::vector<std::thread> pool;
    for (size_t i = 0; i < threads; ++i)
      pool.emplace_back(start_reading);
    for (auto &t : pool)
      t.join();

    return errors.load() ? 0 : std::min(count.load(), size);
  }
};

struct OutputBuffer {
  size_t size;
  size_t pokemon_out_dim;
  size_t active_out_dim;
  size_t side_out_dim;
  py::array_t<float> pokemon;         // (B, 2, 5, pokemon_out_dim)
  py::array_t<float> active;          // (B, 2, 1, active_out_dim)
  py::array_t<float> sides;           // (B, 2, 1, side_out_dim)
  py::array_t<float> value;           // (B, 1)
  py::array_t<float> p1_policy_logit; // (B, policy_out_dim)
  py::array_t<float> p2_policy_logit; // (B, policy_out_dim)
  py::array_t<float> p1_policy;       // (B, 9)
  py::array_t<float> p2_policy;       // (B, 9)

  // last dim is neg inf, invalid actions map to it
  static constexpr size_t policy_out_dim = Encode::Battle::Policy::n_dim + 1;

  OutputBuffer(size_t sz, size_t pod = NN::Battle::Default::pokemon_out_dim,
               size_t aod = NN::Battle::Default::active_out_dim)
      : size(sz), pokemon_out_dim{pod}, active_out_dim{aod},
        side_out_dim{(1 + active_out_dim) + 5 * (1 + pokemon_out_dim)} {
    const auto make_shape = [sz](std::vector<size_t> dims) {
      dims[0] = static_cast<size_t>(sz);
      return dims;
    };
    pokemon = py::array_t<float>(make_shape({0, 2, 5, pokemon_out_dim}));
    active = py::array_t<float>(make_shape({0, 2, 1, active_out_dim}));
    sides = py::array_t<float>(make_shape({0, 2, 1, side_out_dim}));
    value = py::array_t<float>(make_shape({0, 1}));
    p1_policy_logit = py::array_t<float>(make_shape({0, policy_out_dim}));
    p2_policy_logit = py::array_t<float>(make_shape({0, policy_out_dim}));
    p1_policy = py::array_t<float>(make_shape({0, 9}));
    p2_policy = py::array_t<float>(make_shape({0, 9}));
  }

  void clear() {
    std::fill_n(pokemon.mutable_data(), pokemon.size(), 0.0f);
    std::fill_n(active.mutable_data(), active.size(), 0.0f);
    std::fill_n(sides.mutable_data(), sides.size(), 0.0f);
    std::fill_n(value.mutable_data(), value.size(), 0.0f);
    std::fill_n(p1_policy_logit.mutable_data(), p1_policy_logit.size(), 0.0f);
    std::fill_n(p2_policy_logit.mutable_data(), p2_policy_logit.size(), 0.0f);
    std::fill_n(p1_policy.mutable_data(), p1_policy.size(), 0.0f);
    std::fill_n(p2_policy.mutable_data(), p2_policy.size(), 0.0f);
    // Set final logit column to -inf
    auto p1 = p1_policy_logit.mutable_unchecked<2>();
    auto p2 = p2_policy_logit.mutable_unchecked<2>();
    for (size_t i = 0; i < p1.shape(0); ++i) {
      p1(i, p1.shape(1) - 1) = -std::numeric_limits<float>::infinity();
      p2(i, p2.shape(1) - 1) = -std::numeric_limits<float>::infinity();
    }
  }
};

OutputBuffer cpp_inference(std::string network_path,
                           const BattleFrame &battle_frames) {
  OutputBuffer buffer{battle_frames.size};
  NN::Battle::Network network{};
  std::ifstream file{network_path, std::ios::binary};
  if (!file) {
    throw std::runtime_error{"Can't open network"};
  }

  for (auto i = 0; i < output.size; ++i) {
    output.value[i, 0] = 0;
  }
  return buffer;
}

namespace py = pybind11;

PYBIND11_MODULE(pyoak, m) {
  m.doc() = "Python bindings for oak";

  py::class_<RuntimeSearch::Nodes>(m, "Nodes")
      .def(py::init<>())
      .def("reset", &RuntimeSearch::Nodes::reset);

  py::class_<RuntimeSearch::Agent>(m, "Agent")
      .def(py::init<>())
      .def_readwrite("search_budget", &RuntimeSearch::Agent::search_budget)
      .def_readwrite("bandit", &RuntimeSearch::Agent::bandit)
      .def_readwrite("eval", &RuntimeSearch::Agent::eval)
      .def_readwrite("matrix_ucb", &RuntimeSearch::Agent::matrix_ucb)
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
        auto options = PKMN::options();
        pkmn_gen1_chance_options chance_options{};
        chance_options.durations = input.durations;
        PKMN::set(options, chance_options);
        input.result = PKMN::update(input.battle, c1, c2, options);
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
  m.attr("build_policy_hidden_dim") = NN::Build::Default::policy_hidden_dim;
  m.attr("build_value_hidden_dim") = NN::Build::Default::value_hidden_dim;
  m.attr("build_max_actions") = Tensorizer::max_actions;
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

  // Strings
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

  py::class_<BattleFrame>(m, "BattleFrame")
      .def(py::init<size_t>())
      .def("uncompress_from_bytes", &BattleFrame::uncompress_from_bytes)
      .def_static("from_bytes", &BattleFrame::from_bytes)
      .def_readonly("size", &BattleFrame::size);

  py::class_<EncodedBattleFrame>(m, "EncodedBattleFrame")
      .def(py::init<size_t>())
      .def("clear", &EncodedBattleFrame::clear)
      .def("uncompress_from_bytes", &EncodedBattleFrame::uncompress_from_bytes)
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
      .def_readonly("score", &EncodedBattleFrame::score)
      .def("sample", &EncodedBattleFrame::sample);

  py::class_<OutputBuffer>(m, "OutputBuffer")
      .def(py::init<size_t, size_t, size_t>(), py::arg("size"),
           py::arg("pokemon_out_dim") = Encode::Battle::Pokemon::n_dim,
           py::arg("active_out_dim") = Encode::Battle::Active::n_dim)
      .def_readonly("size", &OutputBuffer::size)
      .def_readonly("pokemon_out_dim", &OutputBuffer::pokemon_out_dim)
      .def_readonly("active_out_dim", &OutputBuffer::active_out_dim)
      .def_readonly("pokemon", &OutputBuffer::pokemon)
      .def_readonly("active", &OutputBuffer::active)
      .def_readonly("sides", &OutputBuffer::sides)
      .def_readonly("value", &OutputBuffer::value)
      .def_readonly("p1_policy_logit", &OutputBuffer::p1_policy_logit)
      .def_readonly("p2_policy_logit", &OutputBuffer::p2_policy_logit)
      .def_readonly("p1_policy", &OutputBuffer::p1_policy)
      .def_readonly("p2_policy", &OutputBuffer::p2_policy)
      .def("clear", &OutputBuffer::clear);

  py::class_<SampleIndexer>(m, "SampleIndexer")
      .def(py::init<>())
      .def("get", &SampleIndexer::get)
      .def("prune", &SampleIndexer::prune)
      .def("size", &SampleIndexer::size);

  m.def("read_battle_data", &read_battle_data, py::arg("path"),
        py::arg("max_battles") = 1'000'000);
}

} // namespace
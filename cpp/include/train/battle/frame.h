#pragma once

#include <train/battle/compressed-frame.h>
#include <train/battle/target.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Train {

namespace Battle {

struct Frame {
  uint8_t m, n;
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  std::array<pkmn_choice, 9> p1_choices;
  std::array<pkmn_choice, 9> p2_choices;
  Target target;
};

std::vector<Frame> uncompress(const CompressedFrames &compressed) {

  using policy_type = uint16_t;
  using value_type = uint16_t;
  using offset_type = uint32_t;
  using Offset = offset_type;
  using FrameCount = uint16_t;

  pkmn_gen1_battle b = compressed.battle;
  auto options = PKMN::options();
  pkmn_result result{80};

  std::vector<Frame> frames;
  frames.reserve(compressed.updates.size());

  for (const auto &update : compressed.updates) {
    frames.emplace_back();
    Frame &frame = frames.back();
    frame.m = update.m;
    frame.n = update.n;
    frame.target.iterations = update.iterations;

    const auto [p1_choices, p2_choices] = PKMN::choices(b, result);

    std::memcpy(frame.battle.bytes, b.bytes, PKMN::Layout::Sizes::Battle);
    std::memcpy(frame.durations.bytes, PKMN::durations(options).bytes,
                PKMN::Layout::Sizes::Durations);
    frame.result = result;
    for (int i = 0; i < update.m; ++i) {
      frame.target.p1_empirical[i] =
          uncompress_probs<policy_type, float>(update.p1_empirical[i]);
      frame.target.p1_nash[i] =
          uncompress_probs<policy_type, float>(update.p1_nash[i]);
      frame.p1_choices[i] = p1_choices[i];
    }
    for (int i = 0; i < update.n; ++i) {
      frame.target.p2_empirical[i] =
          uncompress_probs<policy_type, float>(update.p2_empirical[i]);
      frame.target.p2_nash[i] =
          uncompress_probs<policy_type, float>(update.p2_nash[i]);
      frame.p2_choices[i] = p2_choices[i];
    }

    frame.target.empirical_value =
        uncompress_probs<value_type, float>(update.empirical_value);
    frame.target.nash_value =
        uncompress_probs<value_type, float>(update.nash_value);
    frame.target.score = PKMN::score(compressed.result);
    result = PKMN::update(b, update.c1, update.c2, options);
  }

  return frames;
}

struct FrameInput {
  uint8_t *m;
  uint8_t *n;
  uint8_t *battle;
  uint8_t *durations;
  uint8_t *result;
  uint8_t *p1_choices;
  uint8_t *p2_choices;
  uint32_t *iterations;
  float *p1_empirical;
  float *p1_nash;
  float *p2_empirical;
  float *p2_nash;
  float *empirical_value;
  float *nash_value;
  float *score;

  void write(const Frame &frame) {
    *m++ = frame.m;
    *n++ = frame.n;
    std::memcpy(battle, frame.battle.bytes, sizeof(pkmn_gen1_battle));
    battle += sizeof(pkmn_gen1_battle);
    std::memcpy(durations, frame.durations.bytes,
                sizeof(pkmn_gen1_chance_durations));
    durations += sizeof(pkmn_gen1_chance_durations);
    std::memcpy(result, &frame.result, sizeof(pkmn_result));
    result += sizeof(pkmn_result);
    std::fill_n(p1_empirical, 9, 0);
    std::fill_n(p1_nash, 9, 0);
    std::fill_n(p2_empirical, 9, 0);
    std::fill_n(p2_nash, 9, 0);
    for (int i = 0; i < frame.m; ++i) {
      p1_empirical[i] = frame.target.p1_empirical[i];
      p1_nash[i] = frame.target.p1_nash[i];
      p1_choices[i] = frame.p1_choices[i];
    }
    for (int i = 0; i < frame.n; ++i) {
      p2_empirical[i] = frame.target.p2_empirical[i];
      p2_nash[i] = frame.target.p2_nash[i];
      p2_choices[i] = frame.p2_choices[i];
    }
    p1_empirical += 9;
    p1_nash += 9;
    p2_empirical += 9;
    p2_nash += 9;
    p1_choices += 9;
    p2_choices += 9;
    *iterations++ = frame.target.iterations;
    *empirical_value++ = frame.target.empirical_value;
    *nash_value++ = frame.target.nash_value;
    *score++ = frame.target.score;
  }
};

} // namespace Battle

} // namespace Train

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

    const auto frames_vec = uncompress(compressed_frames);
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

#pragma once

#include <py/battle/target.h>
#include <train/battle/compressed-frame.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Py::Battle {

namespace py = pybind11;

namespace {

struct Frames : public Target {
  py::array_t<uint8_t> battle;
  py::array_t<uint8_t> durations;
  py::array_t<uint8_t> result;
  py::array_t<uint8_t> choices;

  Frames(size_t size) : Target{size} {
    battle = py::array_t<uint8_t>(std::vector<size_t>{size, 384});
    durations = py::array_t<uint8_t>(std::vector<size_t>{size, 8});
    result = py::array_t<uint8_t>(std::vector<size_t>{size, 1});
    choices = py::array_t<uint8_t>(std::vector<size_t>{size, 2, 9});
  }

  void write(const auto index, const pkmn_gen1_battle &b,
             const pkmn_gen1_chance_durations &d, pkmn_result r,
             const Train::Battle::CompressedFrames::Update &update,
             float terminal) {
    Py::Battle::Target::write(index, update);
    score.mutable_data()[index] = terminal;
    std::memcpy(battle.mutable_data() + (index * 384), b.bytes, 384);
    std::memcpy(durations.mutable_data() + (index * 8), d.bytes, 8);
    // TODO get choices and write
    result.mutable_data()[index] = r;
  }

  void uncompress_from_bytes(const py::bytes &data) {
    std::string_view sv(data);
    const char *raw_data = sv.data();
    Train::Battle::CompressedFrames compressed_frames{};
    compressed_frames.read(raw_data);
    auto battle = compressed_frames.battle;
    auto options = PKMN::options();
    auto result = PKMN::result();
    const auto score = PKMN::score(compressed_frames.result);
    for (auto i = 0; i < compressed_frames.updates.size(); ++i) {
      const auto &update = compressed_frames.updates[i];
      write(i, battle, PKMN::durations(options), result, update, score);
      result = PKMN::update(battle, update.c1, update.c2, options);
    }
    assert(result == compressed_frames.result);
  }

  static Frames from_bytes(const py::bytes &data, size_t size) {
    Frames f(size);
    f.uncompress_from_bytes(data);
    return f;
  }
};

}

} // namespace Py::Battle
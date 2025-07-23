#include <encode/battle.h>
#include <encode/team.h>
#include <train/compressed-frame.h>
#include <train/frame.h>
#include <util/debug-log.h>

#include <cstdint>
#include <fstream>
#include <iostream>

extern const int pokemon_input_dim = Encode::Pokemon::n_dim;
extern const int active_input_dim = Encode::Active::n_dim;

extern "C" int read_battle_offsets(const char *path, uint16_t *out,
                                   size_t max_count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  std::cout << path << std::endl;

  size_t count = 0;

  auto *buf = new char[100000];

  while (count < max_count) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }

    *buf = {};
    file.seekg(-2, std::ios::cur);
    file.read(buf, offset);
    std::cout << "offset: " << offset << std::endl;
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    DebugLog<256> debug_log{};
    auto battle = battle_frames.battle;
    debug_log.set_header(battle);
    pkmn_gen1_battle_options options{};
    for (const auto &update : battle_frames.updates) {
      debug_log.update(battle, update.c1, update.c2, options);
    }

    debug_log.save_data_to_path(std::to_string(count) + ".log");

    out[count++] = offset;

    if (file.peek() == EOF) {
      return count;
    }
  }

  return static_cast<int>(count);
}

extern "C" int read_buffer_to_frames(const char *path, size_t max_count,
                                     uint8_t *m, uint8_t *n, uint8_t *battle,
                                     uint8_t *durations, uint8_t *result,
                                     uint8_t *p1_choices, uint8_t *p2_choices,
                                     float *p1_empirical, float *p1_nash,
                                     float *p2_empirical, float *p2_nash,
                                     float *empirical_value, float *nash_value,
                                     float *score) {

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  if (!(m && n && battle && durations && result && p1_choices && p2_choices &&
        p1_empirical && p2_empirical && p1_nash && p2_nash && empirical_value &&
        nash_value && score)) {
    std::cerr << "bad ptr" << std::endl;
    return -1;
  }

  int count = 0;

  Train::FrameInput input{.m = m,
                          .n = n,
                          .battle = battle,
                          .durations = durations,
                          .result = result,
                          .p1_choices = p1_choices,
                          .p2_choices = p2_choices,
                          .p1_empirical = p1_empirical,
                          .p1_nash = p1_nash,
                          .p2_empirical = p2_empirical,
                          .p2_nash = p2_nash,
                          .empirical_value = empirical_value,
                          .nash_value = nash_value,
                          .score = score};
  std::cout << "score 2 " << score << std::endl;
  float *buf;
  while (true) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);

    std::cout << offset << std::endl;

    std::vector<char> buffer{};
    buffer.resize(offset);
    auto buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    const auto frames = battle_frames.uncompress();

    for (const auto frame : frames) {
      std::cout << "start" << std::endl;
      input.write(frame);
      std::cout << "end" << std::endl;
    }
    count += frames.size();

    if (file.peek() == EOF) {
      return count;
    }
  }
}

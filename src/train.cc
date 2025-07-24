#include <encode/battle.h>
#include <encode/team.h>
#include <train/compressed-frame.h>
#include <train/frame.h>
#include <util/debug-log.h>

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>

extern "C" const int pokemon_in_dim = Encode::Pokemon::n_dim;
extern "C" const int active_in_dim = Encode::Active::n_dim;

extern "C" int read_battle_offsets(const char *path, uint16_t *out,
                                   size_t max_count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

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

  const auto ptrs = std::bit_cast<std::array<void *, 14>>(input);
  for (auto *x : ptrs) {
    if (x == 0) {
      std::cerr << "bad ptr" << std::endl;
      return -1;
    }
  }

  while (true) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);

    std::vector<char> buffer{};
    buffer.resize(offset);
    auto buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    if (count + battle_frames.updates.size() >= max_count) {
      return count;
    }

    const auto frames = battle_frames.uncompress();

    for (const auto frame : frames) {
      input.write(frame);
    }
    count += frames.size();

    if (file.peek() == EOF) {
      return count;
    }
  }
}

extern "C" int read_encoded_buffer_to_frames(const char *path, size_t max_count,
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

  const auto ptrs = std::bit_cast<std::array<void *, 14>>(input);
  for (auto *x : ptrs) {
    if (x == 0) {
      std::cerr << "bad ptr" << std::endl;
      return -1;
    }
  }

  while (true) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);

    std::vector<char> buffer{};
    buffer.resize(offset);
    auto buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    if (count + battle_frames.updates.size() >= max_count) {
      return count;
    }

    const auto frames = battle_frames.uncompress();

    for (const auto frame : frames) {
      Encode::EncodedFrame encoded{};
      const auto &battle = View::ref(frame.battle);
      const auto &durations = View::ref(frame.durations);

      for (auto s = 0; s < 2; ++s) {
        const auto &side = battle.sides[s];
        const auto &duration = durations.get(s);
        const auto &stored = side.stored();

        Encode::Active::write(stored, side.active, duration,
                              encoded.active[s][0].data());

        for (auto slot = 2; slot <= 6; ++slot) {
          const auto &pokemon = side.get(slot);

          const auto sleep = duration.sleep(slot - 1);
          Encode::Pokemon::write(pokemon, sleep,
                                 encoded.pokemon[s][slot - 2].data());
        }
      }
      
    }
    count += frames.size();

    if (file.peek() == EOF) {
      return count;
    }
  }
}

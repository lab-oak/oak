#include <util/debug-log.h>
#include <nn/encode.h>
#include <train/compressed-frame.h>
#include <train/frame.h>

#include <cstdint>
#include <fstream>
#include <iostream>

namespace {

struct EndcodedData {
  float *pokemon;
  float *active;
};

} // namespace

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
    file.seekg(0, std::ios::beg);
    file.read(buf, offset);
    std::cout << "offset: " << offset << std::endl;
    std::cout << "buf[0 :2] " << (int)buf[0] + 256 * (int)buf[1] << std::endl;
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

    // if (!file.seekg(offset, std::ios::cur)) {
    //   std::cerr << "bad seekg" << std::endl;
    //   return -1;
    // }
    if (file.peek() == EOF) {
      return count;
    }
  }

  return static_cast<int>(count);
}

extern "C" int parse_buffer(const char *path, uint16_t *out, size_t max_count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }
  return 0;
}

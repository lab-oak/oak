#include <nn/encoding.h>
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

  size_t count = 0;

  while (count < max_count) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }

    out[count++] = offset;

    if (!file.seekg(offset - 2, std::ios::cur)) {
      std::cerr << "bad seekg" << std::endl;
      return -1;
    }
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

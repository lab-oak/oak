#include <battle/init.h>
#include <battle/sample-teams.h>
#include <train/compressed-frame.h>
#include <util/random.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <type_traits>

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Input: buffer-path" << std::endl;
    return 1;
  }

  const std::string buffer_path = argv[1];

  std::ifstream file(buffer_path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Could not open file at " << buffer_path << std::endl;
    return 1;
  }

  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  if (!file.read(buffer.data(), file_size)) {
    std::cerr << "Error reading file." << std::endl;
    return 1;
  }

  std::vector<size_t> turns{};
  size_t read_index = 0;
  while (read_index < static_cast<size_t>(file_size)) {
    Train::CompressedFrames<uint16_t, uint16_t> battle_frame{};
    battle_frame.read(buffer.data() + read_index);
    read_index += battle_frame.n_bytes();
    turns.push_back(battle_frame.updates.size());
  }

  std::cout << "turn lengths:" << std::endl;
  for (const auto t : turns) {
    std::cout << t << std::endl;
  }

  std::cout << "read index " << read_index << std::endl;
  std::cout << "file size " << file_size << std::endl;
  assert(read_index == static_cast<size_t>(file_size));
  return 0;
}
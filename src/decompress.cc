#include <battle/init.h>
#include <battle/sample-teams.h>
#include <train/compressed-frame.h>
#include <util/random.h>

#include <iostream>
#include <limits>
#include <type_traits>

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Input: buffer-path" << std::endl;
    return 1;
  }

  const std::string buffer_path = argv[1];
  FILE *file = fopen(buffer_path.data(), "r"); // "r" = read mode

  if (!file) {
    std::cerr << "Could not open file at " << buffer_path << std::endl;
    return 1;
  }

  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  rewind(file);

  std::vector<char> buffer(file_size);
  fread(buffer.data(), 1, file_size, file);

  std::vector<size_t> turns{};
  size_t read_index = 0;
  while (read_index < file_size) {
    Train::CompressedFrames<uint16_t, uint16_t> battle_frame{};
    battle_frame.read(buffer.data() + read_index);
    read_index += battle_frame.n_bytes();
    turns.push_back(battle_frame.updates.size());
  }

  std::cout << "turn lengths:" << std::endl;
  for (const auto t : turns) {
    std::cout << t << std::endl;
  }

  fclose(file);

  std::cout << "read index " << read_index << std::endl;
  std::cout << "file size " << file_size << std::endl;
  assert(read_index == file_size);

  return 0;
}
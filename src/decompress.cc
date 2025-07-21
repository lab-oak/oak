#include <data/sample-teams.h>
#include <libpkmn/pkmn.h>
#include <libpkmn/strings.h>
#include <train/compressed-frame.h>
#include <util/debug-log.h>
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
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buffer.data() + read_index);
    auto battle = battle_frames.battle;
    pkmn_gen1_battle_options options{};
    DebugLog<64> debug_log{};
    debug_log.set_header(battle);
    for (const auto &update : battle_frames.updates) {
      std::cout << Strings::battle_data_to_string(
                       battle,
                       *pkmn_gen1_battle_options_chance_durations(&options), {})
                << '\n';
      std::cout << Strings::side_choice_string(battle.bytes, update.c1) << ' '
                << Strings::side_choice_string(battle.bytes + 184, update.c2)
                << std::endl;

      // auto result = PKMN::update(battle, update.c1, update.c2, options);
      debug_log.update(battle, update.c1, update.c2, options);
    }
    debug_log.save_data_to_path(std::to_string(turns.size()) + ".log");
    read_index += battle_frames.n_bytes();
    turns.push_back(battle_frames.updates.size());
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
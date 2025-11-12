#include <encode/battle/battle.h>
#include <encode/battle/frame.h>
#include <encode/battle/policy.h>
#include <encode/build/trajectory.h>
#include <nn/params.h>
#include <train/battle/compressed-frame.h>
#include <train/battle/frame.h>
#include <train/build/trajectory.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

constexpr auto dim_labels_to_c(const auto &data) {
  constexpr auto size = data.size();
  std::array<const char *, size> ptrs{};
  auto i = 0;
  for (auto &x : data) {
    ptrs[i++] = x.data();
  }
  return ptrs;
}

const auto move_names_ptrs = dim_labels_to_c(PKMN::Data::MOVE_CHAR_ARRAY);
const auto species_names_ptrs = dim_labels_to_c(PKMN::Data::SPECIES_CHAR_ARRAY);
extern "C" const char *const *move_names = move_names_ptrs.data();
extern "C" const char *const *species_names = species_names_ptrs.data();

using Tensorizer = Encode::Build::Tensorizer<>;

extern "C" const auto species_move_list_size =
    static_cast<int>(Tensorizer::species_move_list_size);

consteval auto get_species_move_list_py() {
  std::array<int, species_move_list_size * 2> result{};
  for (auto i = 0; i < species_move_list_size; ++i) {
    const auto pair = Tensorizer::species_move_list(i);
    result[2 * i] = static_cast<int>(pair.first);
    result[2 * i + 1] = static_cast<int>(pair.second);
  }
  return result;
}

const auto species_move_list_py = get_species_move_list_py();

extern "C" const int *species_move_list_ptrs = species_move_list_py.data();

const auto pokemon_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Pokemon::dim_labels);
const auto active_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Active::dim_labels);
const auto policy_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Policy::dim_labels);

extern "C" const char *const *pokemon_dim_labels =
    pokemon_dim_label_ptrs.data();
extern "C" const char *const *active_dim_labels = active_dim_label_ptrs.data();
extern "C" const char *const *policy_dim_labels = policy_dim_label_ptrs.data();

extern "C" const int pokemon_in_dim = Encode::Battle::Pokemon::n_dim;
extern "C" const int active_in_dim = Encode::Battle::Active::n_dim;

extern "C" const int pokemon_hidden_dim = NN::Battle::pokemon_hidden_dim;
extern "C" const int pokemon_out_dim = NN::Battle::pokemon_out_dim;
extern "C" const int active_hidden_dim = NN::Battle::active_hidden_dim;
extern "C" const int active_out_dim = NN::Battle::active_out_dim;
extern "C" const int side_out_dim = NN::Battle::side_out_dim;
extern "C" const int hidden_dim = NN::Battle::hidden_dim;
extern "C" const int value_hidden_dim = NN::Battle::value_hidden_dim;
extern "C" const int policy_hidden_dim = NN::Battle::policy_hidden_dim;
extern "C" const int policy_out_dim = NN::Battle::policy_out_dim;

extern "C" const int build_policy_hidden_dim = NN::Build::policy_hidden_dim;
extern "C" const int build_value_hidden_dim = NN::Build::value_hidden_dim;
// input and output dim for policy net due to encoding
extern "C" const int build_max_actions = Tensorizer::max_actions;

extern "C" int index_compressed_battle_frames(const char *path, char *out_data,
                                              uint16_t *offsets,
                                              uint16_t *frame_counts) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  int n_games = 0;
  size_t total_offset = 0;
  while (true) {

    if (file.peek() == EOF) {
      break;
    }

    uint32_t offset;
    uint16_t frame_count;
    file.read(reinterpret_cast<char *>(&offset), 4);
    if (file.gcount() < 4) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.read(reinterpret_cast<char *>(&frame_count), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad frame count read" << std::endl;
      return -1;
    }
    file.seekg(-6, std::ios::cur);

    std::vector<char> buffer;
    buffer.reserve(offset);
    file.read(buffer.data(), offset);

    if (file.gcount() < offset) {
      std::cerr << "truncated battle frame" << std::endl;
      return -1;
    }

    std::memcpy(out_data + total_offset, buffer.data(), offset);
    offsets[n_games] = offset;
    frame_counts[n_games] = frame_count;
    total_offset += offset;
    ++n_games;
  }

  return n_games;
}

extern "C" void uncompress_training_frames(
    const char *data, uint8_t *m, uint8_t *n, uint8_t *battle,
    uint8_t *durations, uint8_t *result, uint8_t *p1_choices,
    uint8_t *p2_choices, uint32_t *iterations, float *p1_empirical,
    float *p1_nash, float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  Train::Battle::CompressedFrames compressed_frames{};
  compressed_frames.read(data);

  Train::Battle::FrameInput input{.m = m,
                                  .n = n,
                                  .battle = battle,
                                  .durations = durations,
                                  .result = result,
                                  .p1_choices = p1_choices,
                                  .p2_choices = p2_choices,
                                  .iterations = iterations,
                                  .p1_empirical = p1_empirical,
                                  .p1_nash = p1_nash,
                                  .p2_empirical = p2_empirical,
                                  .p2_nash = p2_nash,
                                  .empirical_value = empirical_value,
                                  .nash_value = nash_value,
                                  .score = score};

  const auto frames = compressed_frames.uncompress();
  for (const auto frame : frames) {
    input.write(frame);
  }
}

extern "C" void uncompress_and_encode_training_frames(
    const char *data, uint8_t *m, uint8_t *n, int64_t *p1_choice_indices,
    int64_t *p2_choice_indices, float *pokemon, float *active, float *hp,
    uint32_t *iterations, float *p1_empirical, float *p1_nash,
    float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  Train::Battle::CompressedFrames compressed_frames{};
  compressed_frames.read(data);

  Encode::Battle::FrameInput input{.m = m,
                                   .n = n,
                                   .p1_choice_indices = p1_choice_indices,
                                   .p2_choice_indices = p2_choice_indices,
                                   .pokemon = pokemon,
                                   .active = active,
                                   .hp = hp,
                                   .iterations = iterations,
                                   .p1_empirical = p1_empirical,
                                   .p1_nash = p1_nash,
                                   .p2_empirical = p2_empirical,
                                   .p2_nash = p2_nash,
                                   .empirical_value = empirical_value,
                                   .nash_value = nash_value,
                                   .score = score};

  const auto frames = compressed_frames.uncompress();
  for (const auto frame : frames) {
    Encode::Battle::Frame encoded{frame};
    input.write(encoded, frame.target);
  }
}

extern "C" size_t encode_buffer_multithread(
    const char *const *paths, size_t n_paths, size_t threads, size_t max_count,
    float write_prob, size_t max_game_length, uint8_t *m, uint8_t *n,
    int64_t *p1_choice_indices, int64_t *p2_choice_indices, float *pokemon,
    float *active, float *hp, uint32_t *iterations, float *p1_empirical,
    float *p1_nash, float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  const Encode::Battle::FrameInput input{.m = m,
                                         .n = n,
                                         .p1_choice_indices = p1_choice_indices,
                                         .p2_choice_indices = p2_choice_indices,
                                         .pokemon = pokemon,
                                         .active = active,
                                         .hp = hp,
                                         .iterations = iterations,
                                         .p1_empirical = p1_empirical,
                                         .p1_nash = p1_nash,
                                         .p2_empirical = p2_empirical,
                                         .p2_nash = p2_nash,
                                         .empirical_value = empirical_value,
                                         .nash_value = nash_value,
                                         .score = score};

  const auto ptrs = std::bit_cast<std::array<void *, 15>>(input);
  for (const auto *x : ptrs) {
    if (!x) {
      std::cerr << "encode_buffer: null pointer in input" << std::endl;
      return 0;
    }
  }

  std::atomic<size_t> count{};

  const auto start_reading = [&]() -> void {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_real_distribution<float> dist_real{0, 1};
    std::uniform_int_distribution<size_t> dist_int{0, n_paths - 1};
    std::vector<char> buffer{};

    // read path
    while (true) {

      const auto path_index = dist_int(mt);
      const char *path = paths[path_index];
      std::ifstream file(path, std::ios::binary);
      if (!file) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
      }

      // parse file
      while (true) {

        if (file.peek() == EOF) {
          break;
        }

        // read to buffer
        uint32_t offset;
        uint16_t n_frames;
        file.read(reinterpret_cast<char *>(&offset), 4);
        if (file.gcount() < 4) {
          std::cerr << "Bad offset read" << std::endl;
          return;
        }
        file.read(reinterpret_cast<char *>(&n_frames), 2);
        if (file.gcount() < 2) {
          std::cerr << "Bad n_frames read" << std::endl;
          return;
        }
        file.seekg(-6, std::ios::cur);
        if (offset > 200000) {
          throw std::runtime_error{"Too long"};
          return;
        }
        buffer.resize(offset);

        buffer.clear();
        char *buf = buffer.data();
        file.read(buf, offset);
        // parse buffer to compressed
        Train::Battle::CompressedFrames battle_frames{};
        battle_frames.read(buf);

        if (battle_frames.updates.size() > max_game_length) {
          continue;
        }

        // uncompress
        const auto frames = battle_frames.uncompress();

        // sample, write to encoded_input
        for (const auto frame : frames) {
          const auto r = dist_real(mt);
          if (r >= write_prob) {
            continue;
          }
          const auto cur = count.fetch_add(1);
          if (cur >= max_count) {
            return;
          }
          auto input_correct = input.index(cur);
          Encode::Battle::Frame encoded{frame};
          input_correct.write(encoded, frame.target);
        }
      }
    }
  };

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return std::min(count.load(), max_count);
}

extern "C" int read_build_trajectories(const char *path, int64_t *action,
                                       int64_t *mask, float *policy,
                                       float *value, float *score,
                                       int64_t *start, int64_t *end) {

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  Encode::Build::TrajectoryInput input{.action = action,
                                       .mask = mask,
                                       .policy = policy,
                                       .value = value,
                                       .score = score,
                                       .start = start,
                                       .end = end};

  const auto ptrs = std::bit_cast<std::array<void *, 7>>(input);
  for (const auto *x : ptrs) {
    if (!x) {
      std::cerr << "read_build_trajectories: null pointer in input"
                << std::endl;
      return -1;
    }
  }

  int count = 0;

  while (true) {
    Encode::Build::CompressedTrajectory<> traj;
    file.read(reinterpret_cast<char *>(&traj),
              Encode::Build::CompressedTrajectory<>::size_no_team);
    if (file.gcount() < Encode::Build::CompressedTrajectory<>::size_no_team) {
      std::cerr << "bad build trajectory read" << std::endl;
      return -1;
    }
    const auto format = static_cast<uint8_t>(traj.header.format);
    if (format != 0) {
      std::cerr << "only NoTeam trajectories can be read this way" << std::endl;
      return -1;
    }

    input.write(traj);
    ++count;

    if (file.peek() == EOF) {
      return count;
    }
  }
  return -1;
}

extern "C" void print_battle_data(uint8_t *battle_bytes,
                                  uint8_t *durations_bytes) {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  std::copy(battle_bytes, battle_bytes + PKMN::Layout::Sizes::Battle,
            battle.bytes);
  std::copy(durations_bytes, durations_bytes + PKMN::Layout::Sizes::Durations,
            durations.bytes);
  std::cout << PKMN::battle_data_to_string(battle, durations) << std::endl;
}
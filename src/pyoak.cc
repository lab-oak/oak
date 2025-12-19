#include <encode/battle/battle.h>
#include <encode/battle/frame.h>
#include <encode/battle/policy.h>
#include <encode/build/trajectory.h>
#include <nn/default-hyperparameters.h>
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
const char *const *move_names = move_names_ptrs.data();
const char *const *species_names = species_names_ptrs.data();

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

const int *species_move_list_ptrs = species_move_list_py.data();

const auto pokemon_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Pokemon::dim_labels);
const auto active_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Active::dim_labels);
const auto policy_dim_label_ptrs =
    dim_labels_to_c(Encode::Battle::Policy::dim_labels);

const char *const *pokemon_dim_labels = pokemon_dim_label_ptrs.data();
const char *const *active_dim_labels = active_dim_label_ptrs.data();
const char *const *policy_dim_labels = policy_dim_label_ptrs.data();

extern "C" const int pokemon_in_dim = Encode::Battle::Pokemon::n_dim;
extern "C" const int active_in_dim = Encode::Battle::Active::n_dim;

extern "C" const int pokemon_hidden_dim =
    NN::Battle::Default::pokemon_hidden_dim;
extern "C" const int pokemon_out_dim = NN::Battle::Default::pokemon_out_dim;
extern "C" const int active_hidden_dim = NN::Battle::Default::active_hidden_dim;
extern "C" const int active_out_dim = NN::Battle::Default::active_out_dim;
extern "C" const int side_out_dim = NN::Battle::Default::side_out_dim;
extern "C" const int hidden_dim = NN::Battle::Default::hidden_dim;
extern "C" const int value_hidden_dim = NN::Battle::Default::value_hidden_dim;
extern "C" const int policy_hidden_dim = NN::Battle::Default::policy_hidden_dim;
extern "C" const int policy_out_dim = NN::Battle::Default::policy_out_dim;

extern "C" const int build_policy_hidden_dim =
    NN::Build::Default::policy_hidden_dim;
extern "C" const int build_value_hidden_dim =
    NN::Build::Default::value_hidden_dim;
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

  int n_battles = 0;
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
    offsets[n_battles] = offset;
    frame_counts[n_battles] = frame_count;
    total_offset += offset;
    ++n_battles;
  }

  return n_battles;
}

extern "C" void uncompress_training_frames(
    const char *data, uint8_t *m, uint8_t *n, uint8_t *battle,
    uint8_t *durations, uint8_t *result, uint8_t *p1_choices,
    uint8_t *p2_choices, uint32_t *iterations, float *p1_empirical,
    float *p1_nash, float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score) {

  Train::Battle::CompressedFrames compressed_frames{};
  compressed_frames.read(data);

  using In = Train::Battle::FrameInput;
  In input{.m = m,
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

extern "C" size_t sample_from_battle_data_files(
    size_t max_count, size_t threads, size_t max_battle_length,
    size_t min_interations, // params
    size_t n_paths, const char *const *paths, const int *n_battles,
    const int *const *offsets,
    const int *const *n_frames, // input
    uint8_t *m, uint8_t *n, int64_t *p1_choice_indices,
    int64_t *p2_choice_indices, float *pokemon, float *active, float *hp,
    uint32_t *iterations, float *p1_empirical, float *p1_nash,
    float *p2_empirical, float *p2_nash, float *empirical_value,
    float *nash_value, float *score

) {
  using Input = Encode::Battle::FrameInput;
  Input input{m,
              n,
              p1_choice_indices,
              p2_choice_indices,
              pokemon,
              active,
              hp,
              iterations,
              p1_empirical,
              p1_nash,
              p2_empirical,
              p2_nash,
              empirical_value,
              nash_value,
              score};

  const auto ptrs = std::bit_cast<std::array<void *, 15>>(input);
  if (std::any_of(ptrs.begin(), ptrs.end(), [](const auto x) { return !x; })) {
    std::cerr << "null pointer in input" << std::endl;
    return 0;
  }

  size_t total_battles = 0;
  for (auto i = 0; i < n_paths; ++i) {
    total_battles += n_battles[i];
  }

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() -> void {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist_int{0, total_battles - 1};
    std::vector<char> buffer{};

    const auto report_error = [&](const auto &msg) -> void {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
      return;
    };

    try {

      while (!errors.load()) {

        const auto get_indices = [&]() -> std::pair<uint, uint> {
          auto bi = dist_int(mt);
          auto pi = 0;
          while (bi >= 0) {
            const auto n_battles_in_path = n_battles[pi];
            if (n_battles_in_path > bi) {
              break;
            } else {
              bi -= n_battles_in_path;
              ++pi;
            }
          }
          return {bi, pi};
        };

        const auto [battle_index, path_index] = get_indices();

        if (path_index >= n_paths) {
          return report_error("bad path index");
        }

        const char *path = paths[path_index];
        std::ifstream file(path, std::ios::binary);
        if (!file) {
          return report_error("unable to open file");
        }

        const auto battle_offset = offsets[path_index][battle_index];

        file.seekg(battle_offset, std::ios::beg);

        using Offset = Train::Battle::CompressedFrames::Offset;
        using FrameCount = Train::Battle::CompressedFrames::FrameCount;

        Offset offset;
        FrameCount n_frames;

        file.read(reinterpret_cast<char *>(&offset), sizeof(Offset));
        if (file.gcount() < sizeof(Offset)) {
          return report_error("bad offset read");
        }
        file.read(reinterpret_cast<char *>(&n_frames), sizeof(FrameCount));
        if (file.gcount() < sizeof(FrameCount)) {
          return report_error("bad frame count read");
        }

        if (n_frames > max_battle_length) {
          continue;
        }

        file.seekg(-(sizeof(Offset) + sizeof(FrameCount)), std::ios::cur);
        if ((offset > 200000) || (offset < sizeof(pkmn_gen1_battle))) {
          return report_error("Bad offset length: " + std::to_string(offset) +
                              "; frames: " + std::to_string(n_frames));
        }
        buffer.resize(offset);
        buffer.clear();
        file.read(buffer.data(), offset);
        Train::Battle::CompressedFrames compressed_frames{};
        compressed_frames.read(buffer.data());

        std::vector<int> valid_frame_indices{};
        for (auto i = 0; i < compressed_frames.updates.size(); ++i) {
          const auto &update = compressed_frames.updates[i];
          if (update.iterations >= min_interations) {
            valid_frame_indices.push_back(i);
          }
        }

        if (valid_frame_indices.size() == 0) {
          continue;
        }

        const auto selected_frame_index =
            valid_frame_indices[std::uniform_int_distribution<size_t>{
                0, valid_frame_indices.size() - 1}(mt)];

        const auto frames = compressed_frames.uncompress();
        const auto &frame = frames[selected_frame_index];

        const auto write_index = count.fetch_add(1);
        if (write_index >= max_count) {
          return;
        } else {
          input.index(write_index)
              .write(Encode::Battle::Frame{frame}, frame.target);
        }
      }

    } catch (const std::exception &e) {
      report_error(e.what());
    }
  };

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return errors.load() ? 0 : std::min(count.load(), max_count);
}

extern "C" size_t
read_build_trajectories(size_t max_count, size_t threads, size_t n_paths,
                        const char *const *paths, int64_t *action,
                        int64_t *mask, float *policy, float *value,
                        float *score, int64_t *start, int64_t *end) {

  constexpr auto traj_size =
      Encode::Build::CompressedTrajectory<>::size_no_team;

  using In = Encode::Build::TrajectoryInput<>;
  In input{.action = action,
           .mask = mask,
           .policy = policy,
           .value = value,
           .score = score,
           .start = start,
           .end = end};

  const auto ptrs = std::bit_cast<std::array<void *, 7>>(input);
  if (std::any_of(ptrs.begin(), ptrs.end(), [](const auto x) { return !x; })) {
    std::cerr << "null pointer in input" << std::endl;
    return 0;
  }

  std::atomic<size_t> count{};
  std::atomic<size_t> errors{};

  const auto start_reading = [&]() {
    std::mt19937 mt{std::random_device{}()};
    std::uniform_int_distribution<size_t> file_dist{0, n_paths - 1};

    const auto report_error = [&](const auto &msg) -> void {
      std::cerr << msg << std::endl;
      errors.fetch_add(1);
      return;
    };

    try {
      while (true) {

        const auto path_index = file_dist(mt);
        std::ifstream file(paths[path_index], std::ios::binary);
        if (!file) {
          return report_error("Failed to open file " +
                              std::to_string(path_index));
        }
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0);
        if ((size % traj_size) != 0) {
          return report_error("File " + std::to_string(path_index) + " size " +
                              std::to_string(size) + " is not a multiple of " +
                              std::to_string(traj_size));
        }

        const auto n_trajectories = size / traj_size;

        const auto trajectory_index =
            std::uniform_int_distribution<size_t>{0, n_trajectories - 1}(mt);
        file.seekg(trajectory_index * traj_size);

        Encode::Build::CompressedTrajectory<> traj;
        file.read(reinterpret_cast<char *>(&traj), traj_size);
        if (file.gcount() < traj_size) {
          return report_error("Bad trajectory read");
        }
        const auto format = static_cast<uint8_t>(traj.header.format);
        if (format != 0) {
          return report_error("Only NoTeam trajectories are supported");
        }

        const auto write_index = count.fetch_add(1);
        if (write_index >= max_count) {
          return;
        } else {
          input.index(write_index).write(traj);
        }
      }
    } catch (const std::exception &e) {
      return report_error(e.what());
    }
  };

  std::vector<std::thread> thread_pool{};
  for (auto i = 0; i < threads; ++i) {
    thread_pool.emplace_back(std::thread{start_reading});
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return errors.load() ? 0 : std::min(count.load(), max_count);
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
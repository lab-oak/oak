#include <encode/battle.h>
#include <encode/encoded-frame.h>
#include <encode/policy.h>
#include <encode/team.h>
#include <nn/params.h>
#include <train/compressed-frame.h>
#include <train/frame.h>
#include <util/debug-log.h>

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

auto dim_labels_to_c(const auto &data) {
  constexpr auto size = data.size();
  std::array<const char *, size> ptrs{};
  auto i = 0;
  for (auto &x : data) {
    ptrs[i++] = x.data();
  }
  return ptrs;
}

const auto pokemon_dim_label_ptrs = dim_labels_to_c(Encode::Pokemon::dim_labels);
const auto active_dim_label_ptrs  = dim_labels_to_c(Encode::Active::dim_labels);

extern "C" const char *const *pokemon_dim_labels = pokemon_dim_label_ptrs.data();
extern "C" const char *const *active_dim_labels  = active_dim_label_ptrs.data();

extern "C" const int pokemon_in_dim      = Encode::Pokemon::n_dim;
extern "C" const int active_in_dim       = Encode::Active::n_dim;

extern "C" const int pokemon_hidden_dim  = NN::pokemon_hidden_dim;
extern "C" const int pokemon_out_dim     = NN::pokemon_out_dim;
extern "C" const int active_hidden_dim   = NN::active_hidden_dim;
extern "C" const int active_out_dim      = NN::active_out_dim;
extern "C" const int side_out_dim        = NN::side_out_dim;
extern "C" const int hidden_dim          = NN::hidden_dim;
extern "C" const int value_hidden_dim    = NN::value_hidden_dim;
extern "C" const int policy_hidden_dim   = NN::policy_hidden_dim;
extern "C" const int policy_out_dim      = NN::policy_out_dim;

extern "C" int read_battle_offsets(const char *path, uint16_t *out, size_t max_games) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  int n_games = 0;
  std::vector<char> buffer{};
  while (n_games < max_games) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);
    buffer.resize(offset);
    buffer.clear();
    char *buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);
    out[n_games++] = offset;
    if (file.peek() == EOF) {
      return n_games;
    }
  }
  return n_games;
}

extern "C" int read_buffer_to_frames(const char *path, size_t max_count, float write_prob,
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
  for (const auto *x : ptrs) {
    if (!x) {
      std::cerr << "read_buffer_to_stream: null pointer in input" << std::endl;
      return -1;
    }
  }

  std::mt19937 mt{std::random_device{}()};
  std::uniform_real_distribution<float> dist{0, 1};
  int count = 0;
  std::vector<char> buffer{};
  while (true) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);

    buffer.resize(offset);
    buffer.clear();
    char *buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    const auto frames = battle_frames.uncompress();

    for (const auto frame : frames) {
      const auto r = dist(mt);
      if (r >= write_prob) {
        continue;
      }
      input.write(frame);
      if (++count == max_count) {
        return count;
      }
    }

    if (file.peek() == EOF) {
      return count;
    }
  }
}

extern "C" int encode_buffer(const char *path, size_t max_count, float write_prob,
                             uint8_t *m, uint8_t *n, uint16_t *p1_choice_indices,
                             uint16_t *p2_choice_indices, float *pokemon, float *active,
                             float *hp, float *p1_empirical, float *p1_nash,
                             float *p2_empirical, float *p2_nash,
                             float *empirical_value, float *nash_value,
                             float *score) {

  Encode::EncodedFrameInput input{.m = m,
                                  .n = n,
                                  .p1_choice_indices = p1_choice_indices,
                                  .p2_choice_indices = p2_choice_indices,
                                  .pokemon = pokemon,
                                  .active = active,
                                  .hp = hp,
                                  .p1_empirical = p1_empirical,
                                  .p1_nash = p1_nash,
                                  .p2_empirical = p2_empirical,
                                  .p2_nash = p2_nash,
                                  .empirical_value = empirical_value,
                                  .nash_value = nash_value,
                                  .score = score};

  const auto ptrs = std::bit_cast<std::array<void *, 14>>(input);
  for (const auto *x : ptrs) {
    if (!x) {
      std::cerr << "encode_buffer: null pointer in input" << std::endl;
      return -1;
    }
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return -1;
  }

  std::mt19937 mt{std::random_device{}()};
  std::uniform_real_distribution<float> dist{0, 1};
  int count = 0;
  std::vector<char> buffer{};
  while (true) {
    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);
    if (file.gcount() < 2) {
      std::cerr << "bad offset read" << std::endl;
      return -1;
    }
    file.seekg(-2, std::ios::cur);

    buffer.resize(offset);
    buffer.clear();
    char *buf = buffer.data();
    file.read(buf, offset);
    Train::CompressedFrames<> battle_frames{};
    battle_frames.read(buf);

    const auto frames = battle_frames.uncompress();

    for (const auto frame : frames) {
      const auto r = dist(mt);
      if (r >= write_prob) {
        continue;
      }

      Encode::EncodedFrame encoded{};

      const auto &battle = View::ref(frame.battle);
      const auto &durations = View::ref(frame.durations);
      for (auto s = 0; s < 2; ++s) {
        const auto &side = battle.sides[s];
        const auto &duration = durations.get(s);
        const auto &stored = side.stored();
        encoded.hp[s][0] = (float)stored.hp / stored.stats.hp;
        Encode::Active::write(stored, side.active, duration,
                              encoded.active[s][0].data());

        for (auto slot = 2; slot <= 6; ++slot) {
          const auto &pokemon = side.get(slot);
          const auto sleep = duration.sleep(slot - 1);
          encoded.hp[s][slot - 1] = (float)pokemon.hp / pokemon.stats.hp;
          Encode::Pokemon::write(pokemon, sleep,
                                 encoded.pokemon[s][slot - 2].data());
        }
      }
      encoded.m = frame.m;
      encoded.n = frame.n;
      for (auto i = 0; i < frame.m; ++i) {
        encoded.p1_choice_indices[i] =
            Encode::Policy::get_index(battle.sides[0], frame.p1_choices[i]);
      }
      for (auto i = 0; i < frame.n; ++i) {
        encoded.p2_choice_indices[i] =
            Encode::Policy::get_index(battle.sides[1], frame.p2_choices[i]);
      }
      encoded.target = frame.target;
      input.write(encoded);
      if (++count == max_count) {
        return count;
      }
    }

    if (file.peek() == EOF) {
      return count;
    }
  }
}

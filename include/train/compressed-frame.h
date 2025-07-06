#pragma once

#include <istream>

namespace Train {

template <typename in_type, typename out_type>
constexpr out_type compress_probs(in_type x) {
  if constexpr (std::is_integral_v<out_type>) {
    if constexpr (std::is_signed_v<out_type>) {
      static_assert(false,
                    "Signed integral types not supported to store probs.");
      return {};
    } else {
      return x * std::numeric_limits<out_type>::max();
    }
  } else {
    return static_cast<out_type>(x);
  }
}

template <typename in_type, typename out_type>
constexpr out_type uncompress_probs(in_type x) {
  if constexpr (std::is_integral_v<in_type>) {
    return x / static_cast<out_type>(std::numeric_limits<in_type>::max());
  } else {
    return static_cast<out_type>(x);
  }
}

template <typename policy_type = uint8_t, typename eval_type = uint8_t>
struct CompressedFrames {

  struct Update {
    // rolling out
    uint8_t m, n;
    pkmn_choice c1, c2;
    // training
    eval_type eval;
    std::vector<policy_type> p1_empirical;
    std::vector<policy_type> p1_nash;
    std::vector<policy_type> p2_empirical;
    std::vector<policy_type> p2_nash;

    static constexpr size_t n_bytes_static(auto m, auto n) {
      // The leading byte is the combination m/n
      return 1 + 2 * sizeof(pkmn_result) + sizeof(eval_type) +
             2 * (m + n) * sizeof(policy_type);
    }

    Update() = default;
    Update(const auto &search_output, pkmn_choice c1, pkmn_choice c2)
        : m{static_cast<uint8_t>(search_output.m)},
          n{static_cast<uint8_t>(search_output.n)}, c1{c1}, c2{c2},
          eval{compress_probs<float, eval_type>(search_output.average_value)} {
      p1_empirical.resize(m);
      p1_nash.resize(m);
      p2_empirical.resize(n);
      p2_nash.resize(n);
      for (auto i = 0; i < m; ++i) {
        p1_empirical[i] =
            compress_probs<float, policy_type>(search_output.p1_empirical[i]);
        p1_nash[i] =
            compress_probs<float, policy_type>(search_output.p1_nash[i]);
      }
      for (auto i = 0; i < n; ++i) {
        p2_empirical[i] =
            compress_probs<float, policy_type>(search_output.p1_empirical[i]);
        p2_nash[i] =
            compress_probs<float, policy_type>(search_output.p2_nash[i]);
      }
    }

    size_t n_bytes() const { return n_bytes_static(m, n); }

    bool write(char *buffer) const {
      auto index = 0;
      buffer[index] = (m - 1) + 16 * (n - 1);
      buffer[index + 1] = c1;
      buffer[index + 2] = c2;
      index += 3;
      std::memcpy(buffer + index, reinterpret_cast<const char *>(&eval),
                  sizeof(eval_type));
      index += sizeof(eval_type);
      std::memcpy(buffer + index,
                  reinterpret_cast<const char *>(p1_empirical.data()),
                  m * sizeof(policy_type));
      index += sizeof(policy_type) * m;
      std::memcpy(buffer + index,
                  reinterpret_cast<const char *>(p1_nash.data()),
                  m * sizeof(policy_type));
      index += sizeof(policy_type) * m;
      std::memcpy(buffer + index,
                  reinterpret_cast<const char *>(p2_empirical.data()),
                  n * sizeof(policy_type));
      index += sizeof(policy_type) * n;
      std::memcpy(buffer + index,
                  reinterpret_cast<const char *>(p2_nash.data()),
                  n * sizeof(policy_type));
      // assert(index + sizeof(policy_type) * n == n_bytes());
      return true;
    }

    bool read(const char *buffer) {
      const auto mn = static_cast<uint8_t>(buffer[0]);
      m = (mn % 16) + 1;
      n = (mn / 16) + 1;
      c1 = static_cast<pkmn_choice>(buffer[1]);
      c2 = static_cast<pkmn_choice>(buffer[2]);
      eval = *reinterpret_cast<const eval_type *>(buffer + 3);
      auto index = 3 + sizeof(eval_type);
      p1_empirical.resize(m);
      p1_nash.resize(m);
      p2_empirical.resize(n);
      p2_nash.resize(n);
      std::memcpy(p1_empirical.data(), buffer + index, m * sizeof(policy_type));
      index += m * sizeof(policy_type);
      std::memcpy(p1_nash.data(), buffer + index, m * sizeof(policy_type));
      index += m * sizeof(policy_type);
      std::memcpy(p2_empirical.data(), buffer + index, n * sizeof(policy_type));
      index += n * sizeof(policy_type);
      std::memcpy(p2_nash.data(), buffer + index, n * sizeof(policy_type));
      return true;
    }
  };

  pkmn_gen1_battle battle;
  pkmn_result result;
  std::vector<Update> updates;

  CompressedFrames() = default;
  CompressedFrames(const pkmn_gen1_battle &battle) : battle{battle} {}

  auto n_bytes() const {
    // leading two bytes stores the result of this function
    // this allows us find the starting points of all the CompressedFrames
    // in one large buffer without having to parse them
    auto n = sizeof(uint16_t) + sizeof(pkmn_gen1_battle) + sizeof(pkmn_result);
    for (const auto &update : updates) {
      n += update.n_bytes();
    }
    return n;
  }

  bool write(char *buffer) const {
    reinterpret_cast<uint16_t *>(buffer)[0] = n_bytes();
    auto index = 2;
    std::memcpy(buffer + index, battle.bytes, sizeof(pkmn_gen1_battle));
    index += sizeof(pkmn_gen1_battle);
    buffer[index] = static_cast<char>(result);
    index += 1;
    for (const auto &update : updates) {
      const auto n = update.n_bytes();
      update.write(buffer + index);
      index += n;
    }
    // std::cout << "write; index: " << index << " n_bytes: " << n_bytes()
    //           << std::endl;
    assert(index == n_bytes());
    return true;
  }

  bool read(const char *buffer) {
    const auto buffer_length = *reinterpret_cast<const uint16_t *>(buffer);
    // std::cout << "read; buffer length: " << buffer_length << std::endl;
    std::memcpy(battle.bytes, buffer + 2, sizeof(pkmn_gen1_battle));
    auto index =
        sizeof(uint16_t) + sizeof(pkmn_gen1_battle) + sizeof(pkmn_result);
    while (index < buffer_length) {
      updates.emplace_back();
      auto &update = updates.back();
      update.read(buffer + index);
      index += update.n_bytes();
    }
    // assert(index == buffer_length);
    return true;
  }
};

std::vector<size_t> scan_battle_offsets(std::istream &stream) {
  std::vector<size_t> offsets{};
  return offsets;
}

}; // namespace Train

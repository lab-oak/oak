#pragma once

#include <libpkmn/pkmn.h>

#include <istream>

std::tuple<uint16_t, uint16_t, uint16_t> VERSION = {1, 1, 0};

namespace Train::Battle {

template <typename in_type, typename out_type>
constexpr out_type compress_probs(in_type x) {
  if constexpr (std::is_integral_v<out_type>) {
    if constexpr (std::is_signed_v<out_type>) {
      static_assert(!std::is_same_v<out_type, out_type>,
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

template <typename policy_type = uint16_t, typename value_type = uint16_t,
          typename offset_type = uint32_t>
struct CompressedFramesImpl {

  using Offset = offset_type;
  using FrameCount = uint16_t;

  constexpr static bool new_format{std::is_same_v<offset_type, uint32_t>};

  struct Update {

    using MN = uint8_t;
    using Iter = uint32_t;
    // rolling out
    uint8_t m, n;
    pkmn_choice c1, c2;
    // training
    Iter iterations;
    value_type empirical_value;
    value_type nash_value;
    std::vector<policy_type> p1_empirical;
    std::vector<policy_type> p1_nash;
    std::vector<policy_type> p2_empirical;
    std::vector<policy_type> p2_nash;

    static constexpr size_t n_bytes_static(auto m, auto n) {
      // The leading byte is the combination m/n
      // Max: 1 + 2 + 4 + 4 + 72 = 83
      return sizeof(MN) + 2 * sizeof(pkmn_choice) +
             (new_format ? sizeof(Iter) : 0) + 2 * sizeof(value_type) +
             2 * (m + n) * sizeof(policy_type);
    }

    Update() = default;
    Update(const auto &search_output, pkmn_choice c1, pkmn_choice c2)
        : m{static_cast<uint8_t>(search_output.m)},
          n{static_cast<uint8_t>(search_output.n)}, c1{c1}, c2{c2},
          iterations{static_cast<uint32_t>(search_output.iterations)},
          empirical_value{compress_probs<double, value_type>(
              search_output.empirical_value)},
          nash_value{
              compress_probs<double, value_type>(search_output.nash_value)} {
      p1_empirical.resize(m);
      p1_nash.resize(m);
      p2_empirical.resize(n);
      p2_nash.resize(n);
      for (auto i = 0; i < m; ++i) {
        p1_empirical[i] =
            compress_probs<double, policy_type>(search_output.p1_empirical[i]);
        p1_nash[i] =
            compress_probs<double, policy_type>(search_output.p1_nash[i]);
      }
      for (auto i = 0; i < n; ++i) {
        p2_empirical[i] =
            compress_probs<double, policy_type>(search_output.p2_empirical[i]);
        p2_nash[i] =
            compress_probs<double, policy_type>(search_output.p2_nash[i]);
      }
    }

    size_t n_bytes() const { return n_bytes_static(m, n); }

    bool write(char *buffer) const {
      auto index = 0;
      buffer[index] = (m - 1) | ((n - 1) << 4);
      index += 1;
      buffer[index + 0] = c1;
      buffer[index + 1] = c2;
      index += 2;
      if constexpr (new_format) {
        *reinterpret_cast<Iter *>(buffer + index) = iterations;
        index += sizeof(Iter);
      }
      std::memcpy(buffer + index,
                  reinterpret_cast<const char *>(&empirical_value),
                  sizeof(value_type));
      index += sizeof(value_type);
      std::memcpy(buffer + index, reinterpret_cast<const char *>(&nash_value),
                  sizeof(value_type));
      index += sizeof(value_type);
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
      assert(index + sizeof(policy_type) * n == n_bytes());
      return true;
    }

    bool read(const char *buffer) {
      auto index = 0;
      const auto mn = static_cast<uint8_t>(buffer[index]);
      m = (mn % 16) + 1;
      n = (mn / 16) + 1;
      index += 1;
      c1 = static_cast<pkmn_choice>(buffer[index]);
      index += 1;
      c2 = static_cast<pkmn_choice>(buffer[index]);
      index += 1;
      if constexpr (new_format) {
        iterations = *reinterpret_cast<const Iter *>(buffer + index);
        index += sizeof(Iter);
      } else {
        iterations = 1 >> 12;
      }
      empirical_value = *reinterpret_cast<const value_type *>(buffer + index);
      index += sizeof(value_type);
      nash_value = *reinterpret_cast<const value_type *>(buffer + index);
      index += sizeof(value_type);
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
      assert(index + n * sizeof(policy_type) == n_bytes());
      return true;
    }

    void write_to_tensor(uint8_t *k, Iter *iter, float *empirical_policies,
                         float *nash_policies, float *ev, float *nv) const {
      k[0] = m;
      k[1] = n;
      iter[0] = iterations;
      std::fill_n(empirical_policies, 18, 0);
      std::fill_n(nash_policies, 18, 0);
      for (auto i = 0; i < m; ++i) {
        empirical_policies[0 + i] =
            uncompress_probs<policy_type, float>(p1_empirical[i]);
        nash_policies[0 + i] = uncompress_probs<policy_type, float>(p1_nash[i]);
      }
      for (auto i = 0; i < n; ++i) {
        empirical_policies[9 + i] =
            uncompress_probs<policy_type, float>(p2_empirical[i]);
        nash_policies[9 + i] = uncompress_probs<policy_type, float>(p2_nash[i]);
      }
      ev[0] = uncompress_probs<value_type, float>(empirical_value);
      nv[0] = uncompress_probs<value_type, float>(nash_value);
    }
  };

  pkmn_gen1_battle battle;
  pkmn_result result;
  std::vector<Update> updates;

  CompressedFramesImpl() = default;
  CompressedFramesImpl(const pkmn_gen1_battle &battle) : battle{battle} {}

  uint32_t n_bytes() const {
    uint32_t n = sizeof(Offset) + sizeof(FrameCount) +
                 sizeof(pkmn_gen1_battle) + sizeof(pkmn_result);
    for (const auto &update : updates) {
      n += update.n_bytes();
    }
    return n;
  }

  bool write(char *buffer) const {
    auto index = 0;
    *reinterpret_cast<Offset *>(buffer + index) = n_bytes();
    index += sizeof(Offset);
    *reinterpret_cast<FrameCount *>(buffer + index) = updates.size();
    index += sizeof(FrameCount);
    std::memcpy(buffer + index, battle.bytes, sizeof(pkmn_gen1_battle));
    index += sizeof(pkmn_gen1_battle);
    buffer[index] = static_cast<char>(result);
    index += 1;
    for (const auto &update : updates) {
      const auto n = update.n_bytes();
      update.write(buffer + index);
      index += n;
    }
    assert(index == n_bytes());
    return true;
  }

  bool read(const char *buffer) {
    uint32_t index = 0;
    const Offset buffer_length =
        *reinterpret_cast<const Offset *>(buffer + index);
    index += sizeof(Offset);
    const FrameCount n_updates =
        *reinterpret_cast<const FrameCount *>(buffer + index);
    index += sizeof(FrameCount);
    std::memcpy(battle.bytes, buffer + index, sizeof(pkmn_gen1_battle));
    index += sizeof(pkmn_gen1_battle);
    result = buffer[index];
    index += 1;
    while (index < buffer_length) {
      updates.emplace_back();
      auto &update = updates.back();
      update.read(buffer + index);
      index += update.n_bytes();
    }
    assert(updates.size() == n_updates);
    assert(index == buffer_length);
    return true;
  }
};

using CompressedFrames = CompressedFramesImpl<>;

} // namespace Train::Battle

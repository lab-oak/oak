#pragma once

#include <libpkmn/pkmn.h>
#include <train/battle-frame.h>

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

template <typename policy_type = uint16_t, typename value_type = uint16_t>
struct CompressedBattleFramesImpl {

  struct Update {
    // rolling out
    uint8_t m, n;
    pkmn_choice c1, c2;
    // training
    value_type empirical_value;
    value_type nash_value;
    std::vector<policy_type> p1_empirical;
    std::vector<policy_type> p1_nash;
    std::vector<policy_type> p2_empirical;
    std::vector<policy_type> p2_nash;

    static constexpr size_t n_bytes_static(auto m, auto n) {
      // The leading byte is the combination m/n
      return 1 + 2 * sizeof(pkmn_result) + sizeof(value_type) +
             sizeof(value_type) + 2 * (m + n) * sizeof(policy_type);
    }

    Update() = default;
    Update(const auto &search_output, pkmn_choice c1, pkmn_choice c2)
        : m{static_cast<uint8_t>(search_output.m)},
          n{static_cast<uint8_t>(search_output.n)}, c1{c1}, c2{c2},
          empirical_value{
              compress_probs<float, value_type>(search_output.empirical_value)},
          nash_value{
              compress_probs<float, value_type>(search_output.nash_value)} {
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
            compress_probs<float, policy_type>(search_output.p2_empirical[i]);
        p2_nash[i] =
            compress_probs<float, policy_type>(search_output.p2_nash[i]);
      }
    }

    size_t n_bytes() const { return n_bytes_static(m, n); }

    bool write(char *buffer) const {
      auto index = 0;
      buffer[index] = (m - 1) | ((n - 1) << 4);
      buffer[index + 1] = c1;
      buffer[index + 2] = c2;
      index += 3;
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
      const auto mn = static_cast<uint8_t>(buffer[0]);
      m = (mn % 16) + 1;
      n = (mn / 16) + 1;
      c1 = static_cast<pkmn_choice>(buffer[1]);
      c2 = static_cast<pkmn_choice>(buffer[2]);
      auto index = 3;
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
  };

  pkmn_gen1_battle battle;
  pkmn_result result;
  std::vector<Update> updates;

  CompressedBattleFramesImpl() = default;
  CompressedBattleFramesImpl(const pkmn_gen1_battle &battle) : battle{battle} {}

  auto n_bytes() const {
    auto n =
        2 * sizeof(uint16_t) + sizeof(pkmn_gen1_battle) + sizeof(pkmn_result);
    for (const auto &update : updates) {
      n += update.n_bytes();
    }
    return n;
  }

  bool write(char *buffer) const {
    reinterpret_cast<uint16_t *>(buffer)[0] = n_bytes();
    reinterpret_cast<uint16_t *>(buffer)[1] = updates.size();
    auto index = 4;
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
    const auto buffer_16 = reinterpret_cast<const uint16_t *>(buffer);
    const auto buffer_length = buffer_16[0];
    const auto n_updates = buffer_16[1];
    std::memcpy(battle.bytes, buffer + 4, sizeof(pkmn_gen1_battle));
    auto index = 2 * sizeof(uint16_t) + sizeof(pkmn_gen1_battle);
    result = buffer[index];
    ++index;
    while (index < buffer_length) {
      updates.emplace_back();
      auto &update = updates.back();
      update.read(buffer + index);
      index += update.n_bytes();
    }
    assert(index == buffer_length);
    return true;
  }

  std::vector<BattleFrame> uncompress() const {
    pkmn_gen1_battle b = this->battle;
    auto options = PKMN::options();
    pkmn_result result{80};

    std::vector<BattleFrame> frames;
    frames.reserve(updates.size());

    for (const auto &update : updates) {
      frames.emplace_back();
      BattleFrame &frame = frames.back();
      frame.m = update.m;
      frame.n = update.n;

      const auto [p1_choices, p2_choices] = PKMN::choices(b, result);

      std::memcpy(frame.battle.bytes, b.bytes, Layout::Sizes::Battle);
      std::memcpy(frame.durations.bytes, PKMN::durations(options).bytes,
                  Layout::Sizes::Durations);
      frame.result = result;
      for (int i = 0; i < update.m; ++i) {
        frame.target.p1_empirical[i] =
            uncompress_probs<policy_type, float>(update.p1_empirical[i]);
        frame.target.p1_nash[i] =
            uncompress_probs<policy_type, float>(update.p1_nash[i]);
        frame.p1_choices[i] = p1_choices[i];
      }
      for (int i = 0; i < update.n; ++i) {
        frame.target.p2_empirical[i] =
            uncompress_probs<policy_type, float>(update.p2_empirical[i]);
        frame.target.p2_nash[i] =
            uncompress_probs<policy_type, float>(update.p2_nash[i]);
        frame.p2_choices[i] = p2_choices[i];
      }

      frame.target.empirical_value =
          uncompress_probs<value_type, float>(update.empirical_value);
      frame.target.nash_value =
          uncompress_probs<value_type, float>(update.nash_value);
      frame.target.score = PKMN::score(this->result);
      result = PKMN::update(b, update.c1, update.c2, options);
    }

    return frames;
  }
};

using CompressedBattleFrames = CompressedBattleFramesImpl<>;

}; // namespace Train

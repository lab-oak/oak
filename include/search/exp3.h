

#pragma once

#include <util/int.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <util/to_char.h>
#include <vector>

constexpr float neg_inf = -std::numeric_limits<float>::infinity();

namespace Exp3 {

void softmax(auto &forecast, const auto &gains, float eta) {
  float sum = 0;
  for (auto i = 0; i < 9; ++i) {
    const float y = std::exp(gains[i] * eta);
    forecast[i] = y;
    sum += y;
  }
  for (auto i = 0; i < 9; ++i) {
    forecast[i] /= sum;
  }
}

template <bool enabled> struct JointBanditDataBase;

template <> struct JointBanditDataBase<true> {
  std::array<float, 9> p1_gains;
  std::array<float, 9> p2_gains;
  std::array<uint24_t, 9> p1_visits;
  std::array<uint24_t, 9> p2_visits;
  uint8_t _rows;
  uint8_t _cols;
};

template <> struct JointBanditDataBase<false> {
  std::array<float, 9> p1_gains;
  std::array<float, 9> p2_gains;
  uint8_t _rows;
  uint8_t _cols;
};

#pragma pack(push, 1)
template <float gamma = .1f, bool enable_visits = false>
class JointBanditData : public JointBanditDataBase<enable_visits> {
  static auto consteval get_name() {
    constexpr auto trailing_precision = 3;
    std::array<char, 5 + 3 + trailing_precision> name{"exp3-"};
    auto gamma_char = to_char<gamma, trailing_precision>();
    for (auto i = 0; i < 3 + trailing_precision; ++i) {
      name[i + 5] = gamma_char[i];
    }
    return name;
  }

public:
  static constexpr std::array<char, 11> name = get_name();

  using JointBanditDataBase<enable_visits>::p1_gains;
  using JointBanditDataBase<enable_visits>::p2_gains;
  using JointBanditDataBase<enable_visits>::_rows;
  using JointBanditDataBase<enable_visits>::_cols;

  struct Outcome {
    float p1_value;
    float p2_value;
    float p1_mu;
    float p2_mu;
    uint8_t p1_index;
    uint8_t p2_index;
  };

  void init(auto rows, auto cols) noexcept {
    _rows = rows;
    _cols = cols;
    std::fill(p1_gains.begin(), p1_gains.begin() + rows, 0);
    std::fill(p2_gains.begin(), p2_gains.begin() + cols, 0);
    std::fill(p1_gains.begin() + rows, p1_gains.end(), neg_inf);
    std::fill(p2_gains.begin() + cols, p2_gains.end(), neg_inf);
    if constexpr (enable_visits) {
      std::fill(this->p1_visits.begin(), this->p1_visits.begin() + rows,
                uint24_t{});
      std::fill(this->p2_visits.begin(), this->p2_visits.begin() + cols,
                uint24_t{});
    }
  }

  bool is_init() const noexcept { return this->_rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    if constexpr (enable_visits) {
      ++this->p1_visits[outcome.p1_index];
      ++this->p2_visits[outcome.p2_index];
    }

    if ((p1_gains[outcome.p1_index] += outcome.p1_value / outcome.p1_mu) >= 0) {
      const auto max = p1_gains[outcome.p1_index];
      for (auto &v : p1_gains) {
        v -= max;
      }
    }
    if ((p2_gains[outcome.p2_index] += outcome.p2_value / outcome.p2_mu) >= 0) {
      const auto max = p2_gains[outcome.p2_index];
      for (auto &v : p2_gains) {
        v -= max;
      }
    }
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) const noexcept {
    constexpr float one_minus_gamma = 1 - gamma;
    std::array<float, 9> forecast{};
    if (_rows == 1) {
      outcome.p1_index = 0;
      outcome.p1_mu = 1;
    } else {
      const float eta{gamma / _rows};
      softmax(forecast, p1_gains, eta);
      std::transform(
          forecast.begin(), forecast.end(), forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p1_index = device.sample_pdf(forecast);
      outcome.p1_mu = forecast[outcome.p1_index];
    }
    if (_cols == 1) {
      outcome.p2_index = 0;
      outcome.p2_mu = 1;
    } else {
      const float eta{gamma / _cols};
      softmax(forecast, p2_gains, eta);
      std::transform(
          forecast.begin(), forecast.end(), forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p2_index = device.sample_pdf(forecast);
      outcome.p2_mu = forecast[outcome.p2_index];
    }

    outcome.p1_index =
        std::min(outcome.p1_index, static_cast<uint8_t>(_rows - 1));
    outcome.p2_index =
        std::min(outcome.p2_index, static_cast<uint8_t>(_cols - 1));

    assert(outcome.p1_index < _rows);
    assert(outcome.p2_index < _cols);
  }

  std::string visit_string() const {
    std::stringstream sstream{};
    if constexpr (enable_visits) {
      sstream << "V1: ";
      for (auto i = 0; i < _rows; ++i) {
        sstream << std::to_string(this->p1_visits[i]) << " ";
      }
      sstream << "V2: ";
      for (auto i = 0; i < _cols; ++i) {
        sstream << std::to_string(this->p2_visits[i]) << " ";
      }
      sstream.flush();
    }
    return sstream.str();
  }
};
#pragma pack(pop)

static_assert(sizeof(JointBanditData<.1f, true>) == 128);
static_assert(sizeof(JointBanditData<.1f, false>) == 76);

}; // namespace Exp3

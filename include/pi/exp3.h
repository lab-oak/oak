

#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <immintrin.h>

constexpr float neg_inf = -std::numeric_limits<float>::infinity();

namespace Exp3 {

// inline __m256 exp_ps(__m256 x) {
//   // Save original input for special handling
//   __m256 original_x = x;

//   // Clamp only upper bound (to avoid overflow)
//   x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));

//   // Range reduction: x * log2(e)
//   __m256 fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)); //
//   log2(e) fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |
//   _MM_FROUND_NO_EXC);

//   // Compute exp(x) = 2^n * exp(r)
//   __m256 tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f));
//   __m256 r = _mm256_sub_ps(x, tmp);
//   tmp = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
//   r = _mm256_sub_ps(r, tmp);

//   // Polynomial approximation of exp(r)
//   __m256 r2 = _mm256_mul_ps(r, r);
//   __m256 r3 = _mm256_mul_ps(r2, r);
//   __m256 r4 = _mm256_mul_ps(r3, r);
//   __m256 r5 = _mm256_mul_ps(r4, r);

//   __m256 y = _mm256_fmadd_ps(_mm256_set1_ps(1.9875691500E-4f), r,
//                              _mm256_set1_ps(1.3981999507E-3f));
//   y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073E-3f));
//   y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894E-2f));
//   y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459E-1f));
//   y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201E-1f));
//   y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));

//   // Reconstruct exp(x) = 2^n * exp(r)
//   __m256i emm0 = _mm256_cvttps_epi32(fx);
//   emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
//   emm0 = _mm256_slli_epi32(emm0, 23);
//   __m256 pow2n = _mm256_castsi256_ps(emm0);

//   __m256 result = _mm256_mul_ps(y, pow2n);

//   // Handle -inf manually: exp(-inf) = 0
//   __m256 mask_is_neginf =
//       _mm256_cmp_ps(original_x, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ);
//   result = _mm256_blendv_ps(result, _mm256_setzero_ps(), mask_is_neginf);

//   return result;
// }

// void softmax_simd(float *forecast, const float *gains, float eta) {
//   // Load 8 elements
//   __m256 g_vec = _mm256_loadu_ps(gains);
//   __m256 eta_vec = _mm256_set1_ps(eta);
//   __m256 prod = _mm256_mul_ps(g_vec, eta_vec);
//   __m256 expv = exp_ps(prod);

//   // Store intermediate results
//   _mm256_storeu_ps(forecast, expv);

//   // // Final (9th element)
//   // float final = std::exp(gains[8] * eta);
//   // forecast[8] = final;

//   // Sum
//   __m256 sumv = expv;
//   __m128 lo = _mm256_castps256_ps128(sumv);
//   __m128 hi = _mm256_extractf128_ps(sumv, 1);
//   __m128 sum128 = _mm_add_ps(lo, hi);
//   sum128 = _mm_hadd_ps(sum128, sum128);
//   sum128 = _mm_hadd_ps(sum128, sum128);
//   float sum = _mm_cvtss_f32(sum128);

//   // Normalize
//   __m256 sum_vec = _mm256_set1_ps(sum);
//   __m256 norm = _mm256_div_ps(expv, sum_vec);
//   _mm256_storeu_ps(forecast, norm);
//   // forecast[8] = final / sum;
// }

// void softmax(std::array<float, 9> &forecast, const std::array<float, 9>
// &gains,
//              float eta) {
//   softmax_simd(forecast.data(), gains.data(), eta);
// }

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

#pragma pack(push, 1)
struct uint24_t {
  std::array<uint8_t, 3> _data;

  constexpr uint24_t() = default;

  constexpr uint24_t(uint32_t value) noexcept {
    _data[0] = value & 0xFF;
    _data[1] = (value >> 8) & 0xFF;
    _data[2] = (value >> 16) & 0xFF;
  }

  constexpr operator uint32_t() const noexcept {
    return (static_cast<uint32_t>(_data[0]) |
            (static_cast<uint32_t>(_data[1]) << 8) |
            (static_cast<uint32_t>(_data[2]) << 16));
  }

  constexpr auto value() const noexcept { return static_cast<uint32_t>(*this); }

  constexpr uint24_t &operator++() noexcept {
    uint32_t value = static_cast<uint32_t>(*this) + 1;
    *this = uint24_t(value);
    return *this;
  }
};
#pragma pack(pop)

struct uint24_t_test {
  static_assert(sizeof(uint24_t) == 3);

  static consteval uint24_t overflow() {
    uint24_t x{};
    for (size_t i = 0; i < (1 << 24); ++i) {
      ++x;
    }
    return x;
  }
};

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
public:
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

    // TODO
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

  std::pair<std::vector<float>, std::vector<float>>
  policies(float iterations) const {

    std::vector<float> p1{};
    std::vector<float> p2{};

    p1.resize(_rows);
    p2.resize(_cols);

    if constexpr (enable_visits) {
      for (auto i = 0; i < _rows; ++i) {
        p1[i] = this->p1_visits[i].value() / (iterations - 1);
      }
      for (auto i = 0; i < _cols; ++i) {
        p2[i] = this->p2_visits[i].value() / (iterations - 1);
      }
    }
    return {p1, p2};
  }
};
#pragma pack(pop)

// static_assert(sizeof(JointBanditData<.1f, true>) == 128);
// static_assert(sizeof(JointBanditData<.1f, false>) == 76);

}; // namespace Exp3

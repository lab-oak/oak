#pragma once

#include <cmath>

void softmax(auto *output, const auto *logits, auto k, float eta = 1) {
  float sum = 0;
  for (auto i = 0; i < k; ++i) {
    const float y = std::exp(logits[i] * eta);
    output[i] = y;
    sum += y;
  }
  for (auto i = 0; i < k; ++i) {
    output[i] /= sum;
  }
}

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
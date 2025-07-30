#pragma once

template <float x, size_t n>
// requires(x >= 0 && x < 10.0);
consteval auto to_char() {
  std::array<char, n + 3> result{};
  float y = x;
  int z = y;
  result[0] = '0' + z;
  result[1] = '.';
  for (auto i = 0; i < n; ++i) {
    y *= 10;
    int z = (int)y % 10;
    result[2 + i] = '0' + z;
  }
  return result;
}
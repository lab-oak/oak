#pragma once

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

void print(const auto &data, const bool newline = true) {
  if constexpr (!debug) {
    return;
  }
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

void print_container(const auto &data, const bool newline = true) {
  if constexpr (!debug) {
    return;
  }
  for (const auto &x : data) {
    std::cout << x << ' ';
  }
  if (newline) {
    std::cout << '\n';
  }
}
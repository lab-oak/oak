#pragma once

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

void debug_print(const auto &data, const bool newline = true) {
  if constexpr (!debug) {
    return;
  }
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

void debug_print_container(const auto &data, const bool newline = true) {
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

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

void print_vec(const auto &data, const bool newline = true) {
  for (const auto &x : data) {
    std::cout << x << ' ';
  }
  if (newline) {
    std::cout << '\n';
  }
}
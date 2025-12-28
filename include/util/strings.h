#pragma once

#include <string>
#include <vector>

namespace Parse {

std::vector<std::string> split(const std::string &input, const char delim) {
  std::vector<std::string> result{};
  std::string token{};

  for (char c : input) {
    if (c == delim) {
      if (!token.empty()) {
        result.push_back(token);
      }
      token.clear();
    } else {
      token += c;
    }
  }
  if (!token.empty()) {
    result.push_back(token);
  }
  return result;
}

} // namespace Parse
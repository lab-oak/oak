#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

std::string get_current_datetime() {
  // Get current time rounded to seconds
  auto now = std::chrono::system_clock::now();
  auto now_sec = std::chrono::floor<std::chrono::seconds>(now);
  std::time_t t = std::chrono::system_clock::to_time_t(now_sec);

  // Convert to local time
  std::tm *tm = std::localtime(&t);

  // Format as YYYY-MM-DD-HH:MM:SS
  std::ostringstream oss;
  oss << std::put_time(tm, "%F-%T");
  return oss.str();
}

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
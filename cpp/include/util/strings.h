#pragma once

#include <search/mcts.h>

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

namespace MCTS {

void print_side(const Output::Side &side) {
  constexpr auto label_width = 8;
  auto print_arr = [](const auto &arr, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      std::cout << arr[i] << "  ";
    }
    std::cout << '\n';
  };
  print_arr(side.empirical, side.k);
  print_arr(side.nash, side.k);
  print_arr(side.beta, side.k);
}

auto output_string(const Output &output, const pkmn_gen1_battle &battle,
                   const auto &p1_labels, const auto &p2_labels) {
  constexpr auto label_width = 8;

  std::stringstream ss{};

  auto print_arr = [&ss](const auto &arr, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      ss << std::left << std::fixed << std::setw(label_width)
         << std::setprecision(3) << arr[i] << "  ";
    }
    ss << '\n';
  };

  const auto fix_label = [label_width](auto label) {
    std::stringstream ss{};
    ss << std::left << std::setw(label_width)
       << label.substr(0, label_width - 1);
    return ss.str();
  };

  ss << "Iterations: " << output.iterations
     << ", Time: " << output.duration.count() / 1000.0 << " sec\n";
  ss << "Value: " << std::fixed << std::setprecision(3)
     << output.empirical_value << "\n";

  ss << "\nP1" << std::endl;
  print_arr(p1_labels, output.p1.k);
  print_arr(output.p1.empirical, output.p1.k);
  print_arr(output.p1.nash, output.p1.k);
  print_arr(output.p1.beta, output.p1.k);
  ss << "P2" << std::endl;
  print_arr(p2_labels, output.p2.k);
  print_arr(output.p2.empirical, output.p2.k);
  print_arr(output.p2.nash, output.p2.k);
  print_arr(output.p2.beta, output.p2.k);

  ss << "\nMatrix:\n";
  std::array<char, label_width + 1> col_offset{};
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.p2.k; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.p1.k; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.p2.k; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        double avg = output.value_matrix[i][j] / output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }

  ss << "\nVisits:\n";
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.p2.k; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.p1.k; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.p2.k; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        auto avg = output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }

  return ss.str();
}

std::string output_string(const MCTS::Output &output,
                          const MCTS::Input &input) {

  std::stringstream ss{};

  constexpr auto label_width = 8;
  const auto &battle = input.battle;
  const auto [p1_labels, p2_labels] = PKMN::choice_labels(battle, input.result);

  auto print_arr = [&ss](const auto &arr, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      ss << std::left << std::fixed << std::setw(label_width)
         << std::setprecision(3) << arr[i] << "  ";
    }
    ss << '\n';
  };

  const auto fix_label = [label_width](auto label) {
    std::stringstream ss{};
    ss << std::left << std::setw(label_width)
       << label.substr(0, label_width - 1);
    return ss.str();
  };

  ss << "Iterations: " << output.iterations
     << ", Time: " << output.duration.count() / 1000.0 << " sec\n";
  ss << "Value: " << std::fixed << std::setprecision(3)
     << output.empirical_value << "\n";

  ss << "\nP1" << std::endl;
  print_arr(p1_labels, output.p1.k);
  print_arr(output.p1.empirical, output.p1.k);
  print_arr(output.p1.nash, output.p1.k);
  ss << "P2" << std::endl;
  print_arr(p2_labels, output.p2.k);
  print_arr(output.p2.empirical, output.p2.k);
  print_arr(output.p2.nash, output.p2.k);

  ss << "\nMatrix:\n";
  std::array<char, label_width + 1> col_offset{};
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.p2.k; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.p1.k; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.p2.k; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        double avg = output.value_matrix[i][j] / output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }

  ss << "\nVisits:\n";
  std::fill(col_offset.data(), col_offset.data() + label_width, ' ');
  ss << fix_label(std::string{col_offset.data()}) << ' ';

  for (size_t j = 0; j < output.p2.k; ++j)
    ss << fix_label(p2_labels[j]) << " ";
  ss << "\n";

  for (size_t i = 0; i < output.p1.k; ++i) {
    ss << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.p2.k; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        ss << " ----    ";
      } else {
        auto avg = output.visit_matrix[i][j];
        ss << std::left << std::fixed << std::setw(label_width)
           << std::setprecision(3) << avg << " ";
      }
    }
    ss << '\n';
  }
  return ss.str();
}

} // namespace MCTS
#pragma once

#include <battle/strings.h>
#include <data/layout.h>

#include <string>
#include <vector>

std::vector<std::string> split(const std::string &input, char delim) {
  std::vector<std::string> result;
  std::string token;

  for (char c : input) {
    if (c == delim) {
      if (token.size()) {
        result.push_back(token);
      }
      token.clear();
    } else {
      token += c;
    }
  }

  result.push_back(token); // push the last token
  return result;
}

Init::Set parse_set(const auto &words) {

  Init::Set pokemon{};

  pokemon.species = Strings::string_to_species(words[0]);

  auto n_moves = 0;

  // remaining words after only mandatory one: species
  for (auto i = 1; i < words.size(); ++i) {
    auto &word = words[i];

    if (n_moves < 4) {
      const auto split_move = split(word, ':');
      std::string move = split_move[0];
      uint8_t pp = 0xFF;
      bool move_parse_success = true;
      try {
        pokemon.moves[n_moves] = Strings::string_to_move(move);
      } catch (...) {
        move_parse_success = false;
      }

      if (move_parse_success) {
        if (split_move.size() > 1) {
          pp = std::min(255LL, std::stoll(split_move[1]));
        }
        pokemon.pp[n_moves] = pp;
        ++n_moves;
        continue;
      }
    }

    if (word.find('%') != std::string::npos) {
      auto percent = std::stoll(split(word, '%')[0]);
      pokemon.hp = percent / 100.0;
    } else {
      using Data::Status;
      auto lower = word;
      std::transform(lower.begin(), lower.end(), lower.begin(),
                     [](auto c) { return std::tolower(c); });
      Data::Status status{};
      bool ignore = false;
      if (lower == "par") {
        status = Status::Paralysis;
      } else if (lower == "frz") {
        status = Status::Freeze;
      } else if (lower == "psn") {
        status = Status::Poison;
      } else if (lower == "brn") {
        status = Status::Burn;
      } else if (lower.starts_with("slp")) {
        const auto sleeps = std::stoll(lower.substr(3));
        status = Status::Sleep7;
        pokemon.sleeps = sleeps;
      } else {
        ignore = true;
      }
      if (!ignore) {
        pokemon.status = static_cast<uint8_t>(status);
      }

      if (lower.starts_with("atk")) {
        pokemon.boosts.atk = std::stoi(lower.substr(3));
      } else if (lower.starts_with("def")) {
        pokemon.boosts.def = std::stoi(lower.substr(3));
      } else if (lower.starts_with("spe")) {
        pokemon.boosts.spe = std::stoi(lower.substr(3));
      } else if (lower.starts_with("spc")) {
        pokemon.boosts.spc = std::stoi(lower.substr(3));
      }
    }
  }
  return pokemon;
}

void print_output(const MCTS::Output &output, const pkmn_gen1_battle &battle,
                  const auto &p1_labels, const auto &p2_labels) {

  auto print_strategy = [](const auto *bytes, const auto &choices,
                           const std::array<double, 9> &strat, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      std::string label = Strings::side_choice_string(bytes, choices[i]);
      std::cout << label << ":" << std::fixed << std::setprecision(2)
                << strat[i] << "  ";
    }
    std::cout << '\n';
  };

  constexpr auto label_width = 8;

  const auto fix_label = [label_width](auto label) {
    std::stringstream ss{};
    ss << std::left << std::setw(label_width) << label.substr(0, label_width);
    return ss.str();
  };

  std::cout << "iterations: " << output.iterations
            << ", time: " << output.duration.count() / 1000.0 << " sec\n";
  std::cout << "value: " << std::fixed << std::setprecision(2)
            << output.average_value << "\n";

  std::cout << "P1 emprical - ";
  print_strategy(battle.bytes, output.p1_choices, output.p1_empirical,
                 output.m);
  std::cout << "P1 nash -     ";
  print_strategy(battle.bytes, output.p1_choices, output.p1_nash, output.m);
  std::cout << "P2 emprical - ";
  print_strategy(battle.bytes + Layout::Sizes::Side, output.p2_choices,
                 output.p2_empirical, output.n);
  std::cout << "P2 nash -     ";
  print_strategy(battle.bytes + Layout::Sizes::Side, output.p2_choices,
                 output.p2_nash, output.n);

  std::cout << "\nmatrix:\n      ";
  std::cout << std::setw(label_width + 1);
  for (size_t j = 0; j < output.n; ++j)
    std::cout << fix_label(p2_labels[j]) << " ";
  std::cout << "\n";

  for (size_t i = 0; i < output.m; ++i) {
    std::cout << fix_label(p1_labels[i]) << " ";
    for (size_t j = 0; j < output.n; ++j) {
      if (output.visit_matrix[i][j] == 0) {
        std::cout << "  ----   ";
      } else {
        double avg = output.value_matrix[i][j] / output.visit_matrix[i][j];
        std::cout << std::left << std::fixed << std::setw(label_width)
                  << std::setprecision(2) << avg << " ";
      }
    }
    std::cout << '\n';
  }
}
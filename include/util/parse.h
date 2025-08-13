#pragma once

#include <libpkmn/layout.h>
#include <libpkmn/strings.h>
#include <search/mcts.h>

#include <iomanip>
#include <string>
#include <vector>

namespace Parse {

std::vector<std::string> split(const std::string &input, const char delim) {
  std::vector<std::string> result;
  std::string token;

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
  result.push_back(token);
  return result;
}

// convert vector of strings to a set
PKMN::Set parse_set(const auto &words) {

  PKMN::Set pokemon{};

  pokemon.species = Strings::string_to_species(words[0]);

  auto n_moves = 0;

  // remaining words after only mandatory one: species
  for (auto i = 1; i < words.size(); ++i) {
    const auto &word = words[i];

    if (n_moves < 4) {
      const auto move_pp = split(word, ':');
      std::string move = move_pp[0];
      uint8_t pp = 0xFF;
      // try/catch is necessary since some sets don't have 4 moves
      // and omitting bad moves is a simple way to make search more effective
      Data::Move m;
      bool move_parse_success = true;
      try {
        m = Strings::string_to_move(move);
      } catch (...) {
        move_parse_success = false;
      }

      if (move_parse_success) {
        pokemon.moves[n_moves] = m;
        if (move_pp.size() > 1) {
          pp = std::min(255LL, std::stoll(move_pp[1]));
        }
        pokemon.pp[n_moves] = pp;
        ++n_moves;
        continue;
      }
    }

    if (word.find('%') != std::string::npos) {
      const auto percent = std::stoll(split(word, '%')[0]);
      pokemon.hp = percent / 100.0;
      continue;
    }

    using Data::Status;
    auto lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](auto c) { return std::tolower(c); });
    if (lower == "par") {
      pokemon.status = Status::Paralysis;
    } else if (lower == "frz") {
      pokemon.status = Status::Freeze;
    } else if (lower == "psn") {
      pokemon.status = Status::Poison;
    } else if (lower == "brn") {
      pokemon.status = Status::Burn;
    } else if (lower.starts_with("slp")) {
      const auto sleeps = std::stoll(lower.substr(3));
      pokemon.status = Status::Sleep7;
      pokemon.sleeps = sleeps;
    } else if (lower.starts_with("rst")) {
      const auto hidden = std::stoll(lower.substr(3));
      if (hidden > 3 || hidden == 0) {
        throw std::runtime_error(
            "parse_pokemon(): Invalid sleep duration for rest: " +
            std::to_string(hidden));
      }
      pokemon.status = Data::rest(hidden);
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

    if (lower.starts_with("lvl")) {
      pokemon.level = std::stoll(lower.substr(3));
    }
  }

  return pokemon;
}

std::array<PKMN::Set, 6> parse_side(const std::string &side_string) {
  const auto set_strings = split(side_string, ';');
  if (set_strings.size() > 6) {
    throw std::runtime_error(
        "parse_side(): too many sets (delineated by \';\')");
  }
  std::array<PKMN::Set, 6> side{};
  std::transform(set_strings.begin(), set_strings.end(), side.begin(),
                 [](const auto &string) {
                   const auto words = split(string, ' ');
                   return parse_set(words);
                 });
  return side;
}

std::pair<pkmn_gen1_battle, pkmn_gen1_chance_durations>
parse_battle(const std::string &battle_string, uint64_t seed = 0x123456) {
  const auto side_strings = split(battle_string, '|');
  if (side_strings.size() != 2) {
    throw std::runtime_error(
        "parse_battle(): must have two sides, delineated by \'|\'");
  }
  const auto p1 = parse_side(side_strings[0]);
  const auto p2 = parse_side(side_strings[1]);
  // auto battle = PKMN::battle(p1, p2, seed);
  // const auto durations = PKMN::battle(p1, p2);

  return {PKMN::battle(p1, p2, seed), PKMN::durations(p1, p2)};
}

} // namespace Parse
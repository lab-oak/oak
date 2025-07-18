#pragma once

#include <pkmn.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <data/moves.h>
#include <data/species.h>
#include <data/strings.h>

#include <battle/view.h>

constexpr const uint8_t *get_pokemon_from_slot(const uint8_t *side,
                                               int slot = 1) {
  const auto index = side[175 + slot] - 1;
  return side + 24 * index;
}

namespace Strings {

std::string side_choice_string(const uint8_t *side, pkmn_choice choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 0: {
    return "pass";
  }
  case 1: {
    return move_string(get_pokemon_from_slot(side, 1)[8 + 2 * choice_data]);
  }
  case 2: {
    return species_string(get_pokemon_from_slot(side, choice_data)[21]);
  }
  default: {
    assert(false);
    return "";
  }
  }
}

bool match(const auto &A, const auto &B) {
  return std::equal(
      A.begin(), A.begin() + std::min(A.size(), B.size()), B.begin(), B.end(),
      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

auto find_unique(const auto &container, const auto &value) {
  const auto matches = [&value](const auto &x) { return match(x, value); };
  auto it = std::find_if(container.begin(), container.end(), matches);
  if (it != container.end()) {
    if (auto other = std::find_if(it + 1, container.end(), matches);
        other != container.end()) {
      if (it->size() == value.size()) {
        return it;
      } else if (other->size() == value.size()) {
        return other;
      } else {
        return container.end(); // return end if not unique
      }
    }
  }
  return it;
}

int unique_index(const auto &container, const auto &value) {
  const auto it = find_unique(container, value);
  if (it == container.end()) {
    return -1;
  }
  return std::distance(container.begin(), it);
}

std::string status(const auto status) {
  const auto byte = static_cast<uint8_t>(status);
  if (byte == 0) {
    return "";
  }
  if (byte & 7) {
    if (byte & 128) {
      return "RST";
    } else {
      return "SLP";
    }
  }
  switch (byte) {
  case 0b00001000:
    return "PSN";
  case 0b00010000:
    return "BRN";
  case 0b00100000:
    return "FRZ";
  case 0b01000000:
    return "PAR";
  case 0b10001000:
    return "TOX";
  default:
    assert(false);
    return "";
  };
}

std::string pokemon_to_string(const uint8_t *const data) {
  std::stringstream sstream{};
  sstream << species_string(data[21]);
  if (data[23] != 100) {
    sstream << " (lvl " << (int)data[23] << ")";
  }
  sstream << status(data[20]);
  for (int m = 0; m < 4; ++m) {
    if (data[2 * m + 10] != 0) {
      sstream << move_string(data[2 * m + 10]) << ": " << (int)data[2 * m + 11]
              << " ";
    }
  }
  return sstream.str();
}

std::string volatiles_to_string(const View::Volatiles &vol) {
  std::stringstream ss{};
  if (vol.bide())
    ss << "(bide)";
  if (vol.thrashing())
    ss << "(thrashing)";
  if (vol.multi_hit())
    ss << "(multi-hit)";
  if (vol.flinch())
    ss << "(flinch)";
  if (vol.charging())
    ss << "(charging)";
  if (vol.binding())
    ss << "(binding)";
  if (vol.invulnerable())
    ss << "(invulnerable)";
  if (vol.confusion())
    ss << "(confused)";
  if (vol.mist())
    ss << "(mist)";
  if (vol.focus_energy())
    ss << "(focus-energy)";
  if (vol.substitute())
    ss << "(substitute)";
  if (vol.recharging())
    ss << "(recharging)";
  if (vol.rage())
    ss << "(rage)";
  if (vol.leech_seed())
    ss << "(leech-seed)";
  if (vol.toxic())
    ss << "(toxic)";
  if (vol.light_screen())
    ss << "(light-screen)";
  if (vol.reflect())
    ss << "(reflect)";
  if (vol.transform())
    ss << "(transform)";
  if (vol.confusion_left())
    ss << "(confusion_left: " << (int)vol.confusion_left() << ")";
  if (vol.attacks())
    ss << "(attacks: " << (int)vol.attacks() << ")";
  if (vol.state())
    ss << "(state: " << (int)vol.state() << ")";
  if (vol.substitute_hp())
    ss << "(sub_hp: " << (int)vol.substitute_hp() << ")";
  if (vol.transform_species())
    ss << "(transform: " << species_string(vol.transform_species()) << ")";
  if (vol.disable_left())
    ss << "(disable_left: " << (int)vol.disable_left() << ")";
  if (vol.disable_move())
    ss << "(disable_move: " << (int)vol.disable_move() << ")";
  if (vol.toxic_counter())
    ss << "(toxic_counter: " << (int)vol.toxic_counter() << ")";
  return ss.str();
}

std::string battle_to_string(const pkmn_gen1_battle &battle) {
  std::stringstream ss{};
  const auto &b = View::ref(battle);
  for (auto s = 0; s < 2; ++s) {
    const auto &side = b.side(s);

    for (auto i = 0; i < 6; ++i) {
      const auto slot = side.order()[i];
      if (slot == 0) {
        continue;
      }
      auto &pokemon = side.pokemon(slot - 1);

      if (i == 0) {
        // pass for now
      }

      ss << species_string(pokemon.species()) << ": ";
      const auto hp = pokemon.hp();
      if (hp != 0) {
        ss << pokemon.percent() << "% (" << pokemon.hp() << '/'
           << pokemon.stats().hp() << ") ";
      } else {
        ss << "KO " << std::endl;
        continue;
      }
      const auto st = pokemon.status();
      if (st != Data::Status::None) {
        ss << status(st) << ' ';
      }
      for (auto m = 0; m < 4; ++m) {
        ss << move_string(pokemon.moves()[m].id) << ' ';
      }
      ss << std::endl;
    }
    ss << std::endl;
  }
  return ss.str();
}

std::string battle_data_to_string(const pkmn_gen1_battle &battle,
                                  const pkmn_gen1_chance_durations &durations,
                                  pkmn_result) {
  std::stringstream ss{};
  const auto &b = View::ref(battle);
  for (auto s = 0; s < 2; ++s) {
    const auto &side = b.side(s);
    const auto &duration = View::ref(durations).duration(s);
    const auto &vol = side.active().volatiles();

    for (auto i = 0; i < 6; ++i) {
      const auto id = side.order(i);
      if (id == 0) {
        continue;
      }
      if (i == 0) {
        if (duration.confusion()) {
          ss << "conf: " << static_cast<int>(duration.confusion());
        }
        if (duration.disable()) {
          ss << " disable: " << static_cast<int>(duration.disable());
        }
        if (duration.attacking()) {
          ss << " attacking: " << static_cast<int>(duration.attacking());
        }
        if (duration.binding()) {
          ss << " binding: " << static_cast<int>(duration.binding());
        }
        ss << std::endl;
        ss << volatiles_to_string(vol) << std::endl;
      } else {
        ss << "  ";
      }
      const auto &pokemon = side.pokemon(id - 1);

      ss << species_char_array(pokemon.species()) << ": ";
      const auto hp = pokemon.hp();
      if (hp != 0) {
        ss << pokemon.percent() << "% (" << pokemon.hp() << '/'
           << pokemon.stats().hp() << ") ";
      } else {
        ss << "KO " << std::endl;
        continue;
      }
      const auto st = pokemon.status();
      if (st != Data::Status::None) {
        ss << status(st) << ' ';
      }
      for (auto m = 0; m < 4; ++m) {
        const auto moveslot = pokemon.moves()[m];
        ss << move_char_array(moveslot.id) << ":" << (int)moveslot.pp << ' ';
      }
      ss << std::endl;
    }
    ss << std::endl;
  }
  return ss.str();
}

Data::Species string_to_species(const std::string &str) {
  const int index = unique_index(Data::SPECIES_CHAR_ARRAY, str);
  if (index < 0) {
    throw std::runtime_error{"Could not match string to Species"};
    return Data::Species::None;
  } else {
    return static_cast<Data::Species>(index);
  }
}

Data::Move string_to_move(const std::string &str) {
  const int index = unique_index(Data::MOVE_CHAR_ARRAY, str);
  if (index < 0) {
    throw std::runtime_error{"Could not match string to Move"};
    return Data::Move::None;
  } else {
    return static_cast<Data::Move>(index);
  }
}

} // namespace Strings
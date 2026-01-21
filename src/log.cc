#include <util/random.h>

#include <teams/ou-sample-teams.h>
#include <util/debug-log.h>

#include <iostream>

enum Opcode : uint8_t {
  null = 0x00,
  laststill = 0x01,
  lastmiss = 0x02,
  move = 0x03,
  switch_ = 0x04,
  cant = 0x05,
  faint = 0x06,
  turn = 0x07,
  win = 0x08,
  tie = 0x09,
  damage = 0x0A,
  heal = 0x0B,
  status = 0x0C,
  curestatus = 0x0D,
  boost = 0x0E,
  clearallboost = 0x0F,
  fail = 0x10,
  miss = 0x11,
  hitcount = 0x12,
  prepare = 0x13,
  mustrecharge = 0x14,
  activate = 0x15,
  fieldactivate = 0x16,
  start = 0x17,
  end = 0x18,
  ohko = 0x19,
  crit = 0x1A,
  supereffective = 0x1B,
  resisted = 0x1C,
  immune = 0x1D,
  transform = 0x1E,
  drag = 0x1F,
  item = 0x20,
  enditem = 0x21,
  cureteam = 0x22,
  sethp = 0x23,
  setboost = 0x24,
  copyboost = 0x25,
  sidestart = 0x26,
  sideend = 0x27,
  singlemove = 0x28,
  singleturn = 0x29,
  weather = 0x2A,
};

struct Ident {
  int player; // 1 or 2
  int slot;   // 1..6
};

static Ident decode_ident(uint8_t b) {
  Ident id;
  id.player = (b & 0x80) ? 2 : 1;
  id.slot = (b & 0x07) + 1;
  return id;
}

static std::string ident_to_string(const Ident &id) {
  // Gen 1 singles → always "a"
  return "p" + std::to_string(id.player) + "a";
}

struct Parser {
  const unsigned char *buf;
  size_t pos = 0;

  std::vector<std::string> log;
  std::optional<size_t> last_move_index;

  Parser(const unsigned char *b) : buf(b) {}

  uint8_t peek_u8() const { return buf[pos]; }

  auto read_u8() { return buf[pos++]; }

  uint16_t read_u16() {
    uint16_t lo = buf[pos++];
    uint16_t hi = buf[pos++];
    return lo | (hi << 8);
  }

  void push(const std::string &s) { log.push_back(s); }

  void annotate_last_move(const std::string &suffix) {
    if (last_move_index) {
      log[*last_move_index] += suffix;
      last_move_index.reset();
    }
  }

  void parse() {
    while (true) {
      const auto opcode = static_cast<Opcode>(read_u8());

      switch (opcode) {

      case Opcode::null: {
        return;
      }
      case Opcode::lastmiss: {
        push("|lastmiss");
        break;
      }
      case Opcode::laststill: {
        push("|laststill");
        break;
      }
      case Opcode::move: {

        // Ident id = decode_ident(read_u8());
        // uint8_t move = read_u8();

        // push("|move|" + ident_to_string(id) + "|" + PKMN::move_string(move));
        // last_move_index = log.size() - 1;
        auto source = read_u8();
        auto move = read_u8();
        auto target = read_u8();
        auto reason = read_u8();
        auto from = 0;
        if (reason == 0x02) {
          from = read_u8();
        }
        break;
      }
      case Opcode::switch_: {
        // Ident id = decode_ident(read_u8());
        // uint8_t species = read_u8();
        // uint8_t hp = read_u8();

        auto ident = read_u8();
        auto species = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();

        push("|switch|" + ident_to_string(decode_ident(ident)) + "|" +
             PKMN::species_string(species) + "|" + std::to_string(hp));
        break;
      }

      case Opcode::cant: {
        // Ident id = decode_ident(read_u8());
        // uint8_t reason = read_u8();
        auto ident = read_u8();
        auto reason = read_u8();
        uint8_t move;
        if (reason == 0x05) {
          move = read_u8();
        }
        push("|cant|" + ident_to_string(decode_ident(ident)) + "|" +
             std::to_string(reason));
        break;
      }

      case Opcode::faint: {
        auto ident = read_u8();
        break;
      }

      case Opcode::turn: {
        auto turn = read_u16();
        break;
      }

      case Opcode::win: {
        auto player = read_u8();
        break;
      }

      case Opcode::tie: {
        break;
      }

      case Opcode::damage: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x05) {
          of = read_u8();
        }
        break;
      }

      case Opcode::heal: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x02) {
          of = read_u8();
        }
        break;
      }

      case Opcode::status: {
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();
        uint8_t from;
        if (reason == 0x01) {
          from = read_u8();
        }
        break;
      }

      case Opcode::curestatus: {
        auto ident = read_u8();
        auto status = read_u8();
        auto reason = read_u8();
        break;
      }

      case Opcode::boost: {
        auto ident = read_u8();
        auto reason = read_u8();
        auto num = read_u8();
        break;
      }

      case Opcode::clearallboost: {
        break;
      }

      case Opcode::fail: {
        auto ident = read_u8();
        auto reason = read_u8();
        break;
      }

      case Opcode::miss: {
        auto ident = read_u8();
        break;
      }

      case Opcode::hitcount: {
        auto ident = read_u8();
        auto num = read_u8();
        break;
      }

      case Opcode::prepare: {
        auto ident = read_u8();
        auto move = read_u8();
        break;
      }

      case Opcode::mustrecharge: {
        auto ident = read_u8();
        break;
      }

      case Opcode::activate: {
        auto ident = read_u8();
        auto reason = read_u8();
        break;
      }

      case Opcode::fieldactivate: {
        break;
      }

      case Opcode::start: {
        auto ident = read_u8();
        auto reason = read_u8();
        uint8_t move_type;
        uint8_t of;

        if (reason == 0x09) {
          move_type = read_u8();
        } else if (reason == 0x0A) {
          move_type = read_u8();
        } else if (reason == 0x0B) {
          move_type = read_u8();
        }
        break;
      }

      case Opcode::end: {
        auto ident = read_u8();
        auto reason = read_u8();
        break;
      }

      case Opcode::ohko: {
        break;
      }

      case Opcode::crit: {
        auto ident = read_u8();
        break;
      }

      case Opcode::supereffective: {
        auto ident = read_u8();
        break;
      }

      case Opcode::resisted: {
        auto ident = read_u8();
        break;
      }
      case Opcode::immune: {
        auto ident = read_u8();
        auto reason = read_u8(); // 0x00 none, 0x01 ohko
        break;
      }
      case Opcode::transform: {
        auto source = read_u8();
        auto target = read_u8();
        break;
      }
      case Opcode::drag: {
        auto ident = read_u8();
        auto species = read_u8();
        auto gender = read_u8();
        auto level = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        break;
      }
      case Opcode::item: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        break;
      }
      case Opcode::enditem: {
        auto target = read_u8();
        auto item = read_u8();
        auto source = read_u8();
        break;
      }
      case Opcode::cureteam: {
        auto ident = read_u8();
        break;
      }
      case Opcode::sethp: {
        auto ident = read_u8();
        auto hp = read_u16();
        auto max_hp = read_u16();
        auto status = read_u8();
        auto reason = read_u8();
        break;
      }
      case Opcode::setboost: {
        auto ident = read_u8();
        auto num = read_u8();
        break;
      }
      case Opcode::copyboost: {
        auto source = read_u8();
        auto target = read_u8();
        break;
      }
      case Opcode::sidestart: {
        auto player = read_u8();
        auto reason = read_u8();
        break;
      }
      case Opcode::sideend: {
        auto player = read_u8();
        auto reason = read_u8();
        uint8_t of;
        if (reason == 0x03) {
          of = read_u8();
        }
        break;
      }
      case Opcode::singlemove: {
        auto ident = read_u8();
        auto move = read_u8();
        break;
      }
      case Opcode::singleturn: {
        auto ident = read_u8();
        auto move = read_u8();
        break;
      }
      case Opcode::weather: {
        auto weather = read_u8();
        auto reason = read_u8();
        break;
      }
      default: {
        std::cout << "ERROR: " << std::to_string(opcode) << std::endl;
        assert(false);
      }
      }
    }
  }
};

int rollout_sample_teams_and_stream_debug_log(int argc, char **argv) {
  constexpr size_t log_size{128};
  using Teams::ou_sample_teams;

  if (argc != 4) {
    std::cout << "Usage: provide two sample team indices [0 - "
              << ou_sample_teams.size() << "] and a u64 seed.\n"
              << "Debug log is piped via stdout e.g.:\n"
              << "\t./release/debug-log 0 1 123456 | "
                 "./extern/engine/src/bin/pkmn-debug "
                 "> index.html"
              << std::endl;
    return 1;
  }

  auto p1 = std::atoll(argv[1]);
  auto p2 = std::atoll(argv[2]);
  uint64_t seed = std::atoll(argv[3]);

  if (p1 >= ou_sample_teams.size()) {
    std::cerr << "Invalid index for p1 team." << std::endl;
    return 1;
  }
  if (p2 >= ou_sample_teams.size()) {
    std::cerr << "Invalid index for p2 team." << std::endl;
    return 1;
  }

  auto battle = PKMN::battle(ou_sample_teams[p1], ou_sample_teams[p2], seed);
  pkmn_gen1_battle_options options{};
  std::array<pkmn_choice, 9> choices{};

  DebugLog<log_size> debug_log{};
  debug_log.set_header(battle);
  mt19937 device{seed};

  auto turns = 0;
  pkmn_choice c1{0};
  pkmn_choice c2{0};
  auto result = PKMN::result();
  while (!pkmn_result_type(result)) {
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    c1 = choices[device.random_int(m)];
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    c2 = choices[device.random_int(n)];

    std::cout << PKMN::battle_data_to_string(battle, PKMN::durations(options))
              << std::endl;
    result = debug_log.update(battle, c1, c2, options);
    const auto *buffer = debug_log.frames.back().data();
    Parser p(buffer);
    p.parse();
    for (auto &s : p.log) {
      std::cout << s << "\n";
    }
    std::cout << "___" << std::endl;

    ++turns;
  }

  for (const char c : debug_log.header) {
    // std::cout << c;
  }
  for (const auto &frame : debug_log.frames) {
    // for (const char c : frame) {
    //   std::cout << c;
    // }
  }

  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_stream_debug_log(argc, argv);
}

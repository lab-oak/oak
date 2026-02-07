#pragma once

namespace PKMN::Layout {

namespace Sizes {
constexpr auto Battle = 384;
constexpr auto Side = 184;
constexpr auto Pokemon = 24;
constexpr auto ActivePokemon = 32;
constexpr auto Actions = 16;
constexpr auto Durations = 8;
constexpr auto Summaries = 12;
} // namespace Sizes

namespace Offsets {

namespace Battle {
constexpr auto sides = 0;
constexpr auto turn = 368;
constexpr auto last_damage = 370;
constexpr auto last_moves = 372;
constexpr auto rng = 376;
} // namespace Battle

namespace Side {
constexpr auto pokemon = 0;
constexpr auto active = 144;
constexpr auto order = 176;
constexpr auto last_selected_move = 182;
constexpr auto last_used_move = 183;
} // namespace Side

namespace Pokemon {
constexpr auto stats = 0;
constexpr auto moves = 10;
constexpr auto hp = 18;
constexpr auto status = 20;
constexpr auto species = 21;
constexpr auto types = 22;
constexpr auto level = 23;
} // namespace Pokemon

namespace ActivePokemon {
constexpr auto stats = 0;
constexpr auto species = 10;
constexpr auto types = 11;
constexpr auto boosts = 12;
constexpr auto volatiles = 16;
constexpr auto moves = 24;
} // namespace ActivePokemon

namespace Stats {
constexpr auto hp = 0;
constexpr auto atk = 2;
constexpr auto def = 4;
constexpr auto spe = 6;
constexpr auto spc = 8;
} // namespace Stats

namespace Boosts {
constexpr auto atk = 0;
constexpr auto def = 4;
constexpr auto spe = 8;
constexpr auto spc = 12;
constexpr auto accuracy = 16;
constexpr auto evasion = 20;
} // namespace Boosts

namespace Volatiles {
constexpr auto Bide = 0;
constexpr auto Thrashing = 1;
constexpr auto MultiHit = 2;
constexpr auto Flinch = 3;
constexpr auto Charging = 4;
constexpr auto Binding = 5;
constexpr auto Invulnerable = 6;
constexpr auto Confusion = 7;
constexpr auto Mist = 8;
constexpr auto FocusEnergy = 9;
constexpr auto Substitute = 10;
constexpr auto Recharging = 11;
constexpr auto Rage = 12;
constexpr auto LeechSeed = 13;
constexpr auto Toxic = 14;
constexpr auto LightScreen = 15;
constexpr auto Reflect = 16;
constexpr auto Transform = 17;
constexpr auto confusion = 18;
constexpr auto attacks = 21;
constexpr auto state = 24;
constexpr auto substitute = 40;
constexpr auto transform = 48;
constexpr auto disable_duration = 52;
constexpr auto disable_move = 56;
constexpr auto toxic = 59;
} // namespace Volatiles

namespace Action {
constexpr auto damage = 0;
constexpr auto hit = 8;
constexpr auto critical_hit = 10;
constexpr auto secondary_chance = 12;
constexpr auto speed_tie = 14;
constexpr auto confused = 16;
constexpr auto paralyzed = 18;
constexpr auto duration = 20;
constexpr auto sleep = 24;
constexpr auto confusion = 26;
constexpr auto disable = 29;
constexpr auto attacking = 31;
constexpr auto binding = 33;
constexpr auto move_slot = 36;
constexpr auto pp = 40;
constexpr auto multi_hit = 44;
constexpr auto psywave = 48;
constexpr auto metronome = 56;
} // namespace Action

namespace Duration {
constexpr auto sleeps = 0;
constexpr auto confusion = 18;
constexpr auto disable = 21;
constexpr auto attacking = 25;
constexpr auto binding = 28;
} // namespace Duration

// namespace Damage {
// constexpr auto base = 0;
// constexpr auto final = 2;
// constexpr auto capped = 4;
// } // namespace Damage

} // namespace Offsets

} // namespace PKMN::Layout
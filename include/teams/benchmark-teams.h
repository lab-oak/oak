
#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/pkmn.h>

namespace Teams {

using enum PKMN::Data::Move;
using PKMN::Set;
using PKMN::Team;
using PKMN::Data::Species;

constexpr std::array<Team, 2> benchmark_teams{
    Team{Set{Species::Jynx, {Blizzard, LovelyKiss, Psychic, Rest}},
         Set{Species::Chansey, {IceBeam, Sing, SoftBoiled, Thunderbolt}},
         Set{Species::Cloyster, {Blizzard, Clamp, Explosion, HyperBeam}},
         Set{Species::Rhydon, {BodySlam, Earthquake, RockSlide, Substitute}},
         Set{Species::Starmie, {Blizzard, Recover, Thunderbolt, ThunderWave}},
         Set{Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}},
    Team{Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
         Set{Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
         Set{Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
         Set{Species::Lapras, {Blizzard, HyperBeam, Sing, Thunderbolt}},
         Set{Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
         Set{Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}},
};

} // namespace Teams
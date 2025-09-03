
#pragma once

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>
#include <libpkmn/pkmn.h>

namespace Teams {

using enum PKMN::Data::Move;
using PKMN::Set;
using PKMN::Team;
using PKMN::Data::Species;

constexpr std::array<Team, 16> ou_sample_teams{
    Team{
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, StunSpore, Explosion}},
        Set{Species::Rhydon, {Earthquake, BodySlam, Substitute, TailWhip}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Starmie, {Blizzard, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, MegaDrain, Explosion}},
        Set{Species::Zapdos, {Thunderbolt, DrillPeck, ThunderWave, Agility}},
        Set{Species::Chansey, {IceBeam, Thunderbolt, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Starmie, {Blizzard, Psychic, ThunderWave, Recover}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, DoubleEdge, Explosion}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {IceBeam, Thunderbolt, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Jynx, {LovelyKiss, Blizzard, Psychic, Rest}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {IceBeam, Counter, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Jynx, {LovelyKiss, Blizzard, Psychic, Rest}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Zapdos, {Thunderbolt, DrillPeck, ThunderWave, Agility}},
        Set{Species::Chansey, {IceBeam, Counter, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Jynx, {LovelyKiss, Blizzard, Psychic, Rest}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Rhydon, {Earthquake, BodySlam, Substitute, TailWhip}},
        Set{Species::Chansey, {IceBeam, Thunderbolt, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Jynx, {LovelyKiss, Blizzard, Psychic, Rest}},
        Set{Species::Cloyster, {Blizzard, Clamp, Explosion, Rest}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {IceBeam, Thunderbolt, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Starmie, {Blizzard, Psychic, ThunderWave, Recover}},
        Set{Species::Cloyster, {Blizzard, Clamp, Explosion, Rest}},
        Set{Species::Jolteon, {Thunderbolt, DoubleKick, ThunderWave, Rest}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, FireBlast}},
    },
    Team{
        Set{Species::Starmie, {Blizzard, Psychic, ThunderWave, Recover}},
        Set{Species::Cloyster, {Blizzard, Clamp, Explosion, Rest}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Rhydon, {Earthquake, BodySlam, Substitute, RockSlide}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, IceBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, MegaDrain, Explosion}},
        Set{Species::Slowbro, {Amnesia, Surf, ThunderWave, Rest}},
        Set{Species::Chansey, {SeismicToss, Reflect, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Thunderbolt}},
    },
    Team{
        Set{Species::Gengar, {Hypnosis, Psychic, Thunderbolt, Explosion}},
        Set{Species::Cloyster, {Blizzard, Clamp, Explosion, Rest}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Gengar, {Hypnosis, Psychic, Thunderbolt, Explosion}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, StunSpore, Explosion}},
        Set{Species::Cloyster, {Blizzard, Clamp, Rest, Explosion}},
        Set{Species::Chansey, {SeismicToss, Reflect, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Thunderbolt}},
    },
    Team{
        Set{Species::Gengar, {Hypnosis, Psychic, Thunderbolt, Explosion}},
        Set{Species::Exeggutor, {SleepPowder, Psychic, StunSpore, Explosion}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Chansey, {IceBeam, Thunderbolt, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Gengar, {Hypnosis, NightShade, Thunderbolt, Explosion}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Zapdos, {Thunderbolt, DrillPeck, ThunderWave, Agility}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, Earthquake, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
    Team{
        Set{Species::Gengar, {Hypnosis, Psychic, Thunderbolt, Explosion}},
        Set{Species::Starmie, {Surf, Thunderbolt, ThunderWave, Recover}},
        Set{Species::Alakazam, {Psychic, SeismicToss, ThunderWave, Recover}},
        Set{Species::Chansey, {Sing, IceBeam, ThunderWave, SoftBoiled}},
        Set{Species::Snorlax, {BodySlam, Reflect, HyperBeam, Rest}},
        Set{Species::Tauros, {BodySlam, HyperBeam, Blizzard, Earthquake}},
    },
};

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
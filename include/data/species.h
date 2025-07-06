#pragma once

#include <data/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Data {
enum class Species : std::underlying_type_t<std::byte> {
  None,
  Bulbasaur,
  Ivysaur,
  Venusaur,
  Charmander,
  Charmeleon,
  Charizard,
  Squirtle,
  Wartortle,
  Blastoise,
  Caterpie,
  Metapod,
  Butterfree,
  Weedle,
  Kakuna,
  Beedrill,
  Pidgey,
  Pidgeotto,
  Pidgeot,
  Rattata,
  Raticate,
  Spearow,
  Fearow,
  Ekans,
  Arbok,
  Pikachu,
  Raichu,
  Sandshrew,
  Sandslash,
  NidoranF,
  Nidorina,
  Nidoqueen,
  NidoranM,
  Nidorino,
  Nidoking,
  Clefairy,
  Clefable,
  Vulpix,
  Ninetales,
  Jigglypuff,
  Wigglytuff,
  Zubat,
  Golbat,
  Oddish,
  Gloom,
  Vileplume,
  Paras,
  Parasect,
  Venonat,
  Venomoth,
  Diglett,
  Dugtrio,
  Meowth,
  Persian,
  Psyduck,
  Golduck,
  Mankey,
  Primeape,
  Growlithe,
  Arcanine,
  Poliwag,
  Poliwhirl,
  Poliwrath,
  Abra,
  Kadabra,
  Alakazam,
  Machop,
  Machoke,
  Machamp,
  Bellsprout,
  Weepinbell,
  Victreebel,
  Tentacool,
  Tentacruel,
  Geodude,
  Graveler,
  Golem,
  Ponyta,
  Rapidash,
  Slowpoke,
  Slowbro,
  Magnemite,
  Magneton,
  Farfetchd,
  Doduo,
  Dodrio,
  Seel,
  Dewgong,
  Grimer,
  Muk,
  Shellder,
  Cloyster,
  Gastly,
  Haunter,
  Gengar,
  Onix,
  Drowzee,
  Hypno,
  Krabby,
  Kingler,
  Voltorb,
  Electrode,
  Exeggcute,
  Exeggutor,
  Cubone,
  Marowak,
  Hitmonlee,
  Hitmonchan,
  Lickitung,
  Koffing,
  Weezing,
  Rhyhorn,
  Rhydon,
  Chansey,
  Tangela,
  Kangaskhan,
  Horsea,
  Seadra,
  Goldeen,
  Seaking,
  Staryu,
  Starmie,
  MrMime,
  Scyther,
  Jynx,
  Electabuzz,
  Magmar,
  Pinsir,
  Tauros,
  Magikarp,
  Gyarados,
  Lapras,
  Ditto,
  Eevee,
  Vaporeon,
  Jolteon,
  Flareon,
  Porygon,
  Omanyte,
  Omastar,
  Kabuto,
  Kabutops,
  Aerodactyl,
  Snorlax,
  Articuno,
  Zapdos,
  Moltres,
  Dratini,
  Dragonair,
  Dragonite,
  Mewtwo,
  Mew
};

static_assert(sizeof(Species) == 1);

struct SpeciesData {
  std::array<uint8_t, 5> base_stats;
  std::array<Types, 2> types;
};

constexpr std::array<SpeciesData, 151> SPECIES_DATA{
    // Bulbasaur
    SpeciesData{
        {45, 49, 49, 45, 65},
        {Types::Grass, Types::Poison},
    },
    // Ivysaur
    {
        {60, 62, 63, 60, 80},
        {Types::Grass, Types::Poison},
    },
    // Venusaur
    {
        {80, 82, 83, 80, 100},
        {Types::Grass, Types::Poison},
    },
    // Charmander
    {
        {39, 52, 43, 65, 50},
        {Types::Fire, Types::Fire},
    },
    // Charmeleon
    {
        {58, 64, 58, 80, 65},
        {Types::Fire, Types::Fire},
    },
    // Charizard
    {
        {78, 84, 78, 100, 85},
        {Types::Fire, Types::Flying},
    },
    // Squirtle
    {
        {44, 48, 65, 43, 50},
        {Types::Water, Types::Water},
    },
    // Wartortle
    {
        {59, 63, 80, 58, 65},
        {Types::Water, Types::Water},
    },
    // Blastoise
    {
        {79, 83, 100, 78, 85},
        {Types::Water, Types::Water},
    },
    // Caterpie
    {
        {45, 30, 35, 45, 20},
        {Types::Bug, Types::Bug},
    },
    // Metapod
    {
        {50, 20, 55, 30, 25},
        {Types::Bug, Types::Bug},
    },
    // Butterfree
    {
        {60, 45, 50, 70, 80},
        {Types::Bug, Types::Flying},
    },
    // Weedle
    {
        {40, 35, 30, 50, 20},
        {Types::Bug, Types::Poison},
    },
    // Kakuna
    {
        {45, 25, 50, 35, 25},
        {Types::Bug, Types::Poison},
    },
    // Beedrill
    {
        {65, 80, 40, 75, 45},
        {Types::Bug, Types::Poison},
    },
    // Pidgey
    {
        {40, 45, 40, 56, 35},
        {Types::Normal, Types::Flying},
    },
    // Pidgeotto
    {
        {63, 60, 55, 71, 50},
        {Types::Normal, Types::Flying},
    },
    // Pidgeot
    {
        {83, 80, 75, 91, 70},
        {Types::Normal, Types::Flying},
    },
    // Rattata
    {
        {30, 56, 35, 72, 25},
        {Types::Normal, Types::Normal},
    },
    // Raticate
    {
        {55, 81, 60, 97, 50},
        {Types::Normal, Types::Normal},
    },
    // Spearow
    {
        {40, 60, 30, 70, 31},
        {Types::Normal, Types::Flying},
    },
    // Fearow
    {
        {65, 90, 65, 100, 61},
        {Types::Normal, Types::Flying},
    },
    // Ekans
    {
        {35, 60, 44, 55, 40},
        {Types::Poison, Types::Poison},
    },
    // Arbok
    {
        {60, 85, 69, 80, 65},
        {Types::Poison, Types::Poison},
    },
    // Pikachu
    {
        {35, 55, 30, 90, 50},
        {Types::Electric, Types::Electric},
    },
    // Raichu
    {
        {60, 90, 55, 100, 90},
        {Types::Electric, Types::Electric},
    },
    // Sandshrew
    {
        {50, 75, 85, 40, 30},
        {Types::Ground, Types::Ground},
    },
    // Sandslash
    {
        {75, 100, 110, 65, 55},
        {Types::Ground, Types::Ground},
    },
    // NidoranF
    {
        {55, 47, 52, 41, 40},
        {Types::Poison, Types::Poison},
    },
    // Nidorina
    {
        {70, 62, 67, 56, 55},
        {Types::Poison, Types::Poison},
    },
    // Nidoqueen
    {
        {90, 82, 87, 76, 75},
        {Types::Poison, Types::Ground},
    },
    // NidoranM
    {
        {46, 57, 40, 50, 40},
        {Types::Poison, Types::Poison},
    },
    // Nidorino
    {
        {61, 72, 57, 65, 55},
        {Types::Poison, Types::Poison},
    },
    // Nidoking
    {
        {81, 92, 77, 85, 75},
        {Types::Poison, Types::Ground},
    },
    // Clefairy
    {
        {70, 45, 48, 35, 60},
        {Types::Normal, Types::Normal},
    },
    // Clefable
    {
        {95, 70, 73, 60, 85},
        {Types::Normal, Types::Normal},
    },
    // Vulpix
    {
        {38, 41, 40, 65, 65},
        {Types::Fire, Types::Fire},
    },
    // Ninetales
    {
        {73, 76, 75, 100, 100},
        {Types::Fire, Types::Fire},
    },
    // Jigglypuff
    {
        {115, 45, 20, 20, 25},
        {Types::Normal, Types::Normal},
    },
    // Wigglytuff
    {
        {140, 70, 45, 45, 50},
        {Types::Normal, Types::Normal},
    },
    // Zubat
    {
        {40, 45, 35, 55, 40},
        {Types::Poison, Types::Flying},
    },
    // Golbat
    {
        {75, 80, 70, 90, 75},
        {Types::Poison, Types::Flying},
    },
    // Oddish
    {
        {45, 50, 55, 30, 75},
        {Types::Grass, Types::Poison},
    },
    // Gloom
    {
        {60, 65, 70, 40, 85},
        {Types::Grass, Types::Poison},
    },
    // Vileplume
    {
        {75, 80, 85, 50, 100},
        {Types::Grass, Types::Poison},
    },
    // Paras
    {
        {35, 70, 55, 25, 55},
        {Types::Bug, Types::Grass},
    },
    // Parasect
    {
        {60, 95, 80, 30, 80},
        {Types::Bug, Types::Grass},
    },
    // Venonat
    {
        {60, 55, 50, 45, 40},
        {Types::Bug, Types::Poison},
    },
    // Venomoth
    {
        {70, 65, 60, 90, 90},
        {Types::Bug, Types::Poison},
    },
    // Diglett
    {
        {10, 55, 25, 95, 45},
        {Types::Ground, Types::Ground},
    },
    // Dugtrio
    {
        {35, 80, 50, 120, 70},
        {Types::Ground, Types::Ground},
    },
    // Meowth
    {
        {40, 45, 35, 90, 40},
        {Types::Normal, Types::Normal},
    },
    // Persian
    {
        {65, 70, 60, 115, 65},
        {Types::Normal, Types::Normal},
    },
    // Psyduck
    {
        {50, 52, 48, 55, 50},
        {Types::Water, Types::Water},
    },
    // Golduck
    {
        {80, 82, 78, 85, 80},
        {Types::Water, Types::Water},
    },
    // Mankey
    {
        {40, 80, 35, 70, 35},
        {Types::Fighting, Types::Fighting},
    },
    // Primeape
    {
        {65, 105, 60, 95, 60},
        {Types::Fighting, Types::Fighting},
    },
    // Growlithe
    {
        {55, 70, 45, 60, 50},
        {Types::Fire, Types::Fire},
    },
    // Arcanine
    {
        {90, 110, 80, 95, 80},
        {Types::Fire, Types::Fire},
    },
    // Poliwag
    {
        {40, 50, 40, 90, 40},
        {Types::Water, Types::Water},
    },
    // Poliwhirl
    {
        {65, 65, 65, 90, 50},
        {Types::Water, Types::Water},
    },
    // Poliwrath
    {
        {90, 85, 95, 70, 70},
        {Types::Water, Types::Fighting},
    },
    // Abra
    {
        {25, 20, 15, 90, 105},
        {Types::Psychic, Types::Psychic},
    },
    // Kadabra
    {
        {40, 35, 30, 105, 120},
        {Types::Psychic, Types::Psychic},
    },
    // Alakazam
    {
        {55, 50, 45, 120, 135},
        {Types::Psychic, Types::Psychic},
    },
    // Machop
    {
        {70, 80, 50, 35, 35},
        {Types::Fighting, Types::Fighting},
    },
    // Machoke
    {
        {80, 100, 70, 45, 50},
        {Types::Fighting, Types::Fighting},
    },
    // Machamp
    {
        {90, 130, 80, 55, 65},
        {Types::Fighting, Types::Fighting},
    },
    // Bellsprout
    {
        {50, 75, 35, 40, 70},
        {Types::Grass, Types::Poison},
    },
    // Weepinbell
    {
        {65, 90, 50, 55, 85},
        {Types::Grass, Types::Poison},
    },
    // Victreebel
    {
        {80, 105, 65, 70, 100},
        {Types::Grass, Types::Poison},
    },
    // Tentacool
    {
        {40, 40, 35, 70, 100},
        {Types::Water, Types::Poison},
    },
    // Tentacruel
    {
        {80, 70, 65, 100, 120},
        {Types::Water, Types::Poison},
    },
    // Geodude
    {
        {40, 80, 100, 20, 30},
        {Types::Rock, Types::Ground},
    },
    // Graveler
    {
        {55, 95, 115, 35, 45},
        {Types::Rock, Types::Ground},
    },
    // Golem
    {
        {80, 110, 130, 45, 55},
        {Types::Rock, Types::Ground},
    },
    // Ponyta
    {
        {50, 85, 55, 90, 65},
        {Types::Fire, Types::Fire},
    },
    // Rapidash
    {
        {65, 100, 70, 105, 80},
        {Types::Fire, Types::Fire},
    },
    // Slowpoke
    {
        {90, 65, 65, 15, 40},
        {Types::Water, Types::Psychic},
    },
    // Slowbro
    {
        {95, 75, 110, 30, 80},
        {Types::Water, Types::Psychic},
    },
    // Magnemite
    {
        {25, 35, 70, 45, 95},
        {Types::Electric, Types::Electric},
    },
    // Magneton
    {
        {50, 60, 95, 70, 120},
        {Types::Electric, Types::Electric},
    },
    // Farfetchd
    {
        {52, 65, 55, 60, 58},
        {Types::Normal, Types::Flying},
    },
    // Doduo
    {
        {35, 85, 45, 75, 35},
        {Types::Normal, Types::Flying},
    },
    // Dodrio
    {
        {60, 110, 70, 100, 60},
        {Types::Normal, Types::Flying},
    },
    // Seel
    {
        {65, 45, 55, 45, 70},
        {Types::Water, Types::Water},
    },
    // Dewgong
    {
        {90, 70, 80, 70, 95},
        {Types::Water, Types::Ice},
    },
    // Grimer
    {
        {80, 80, 50, 25, 40},
        {Types::Poison, Types::Poison},
    },
    // Muk
    {
        {105, 105, 75, 50, 65},
        {Types::Poison, Types::Poison},
    },
    // Shellder
    {
        {30, 65, 100, 40, 45},
        {Types::Water, Types::Water},
    },
    // Cloyster
    {
        {50, 95, 180, 70, 85},
        {Types::Water, Types::Ice},
    },
    // Gastly
    {
        {30, 35, 30, 80, 100},
        {Types::Ghost, Types::Poison},
    },
    // Haunter
    {
        {45, 50, 45, 95, 115},
        {Types::Ghost, Types::Poison},
    },
    // Gengar
    {
        {60, 65, 60, 110, 130},
        {Types::Ghost, Types::Poison},
    },
    // Onix
    {
        {35, 45, 160, 70, 30},
        {Types::Rock, Types::Ground},
    },
    // Drowzee
    {
        {60, 48, 45, 42, 90},
        {Types::Psychic, Types::Psychic},
    },
    // Hypno
    {
        {85, 73, 70, 67, 115},
        {Types::Psychic, Types::Psychic},
    },
    // Krabby
    {
        {30, 105, 90, 50, 25},
        {Types::Water, Types::Water},
    },
    // Kingler
    {
        {55, 130, 115, 75, 50},
        {Types::Water, Types::Water},
    },
    // Voltorb
    {
        {40, 30, 50, 100, 55},
        {Types::Electric, Types::Electric},
    },
    // Electrode
    {
        {60, 50, 70, 140, 80},
        {Types::Electric, Types::Electric},
    },
    // Exeggcute
    {
        {60, 40, 80, 40, 60},
        {Types::Grass, Types::Psychic},
    },
    // Exeggutor
    {
        {95, 95, 85, 55, 125},
        {Types::Grass, Types::Psychic},
    },
    // Cubone
    {
        {50, 50, 95, 35, 40},
        {Types::Ground, Types::Ground},
    },
    // Marowak
    {
        {60, 80, 110, 45, 50},
        {Types::Ground, Types::Ground},
    },
    // Hitmonlee
    {
        {50, 120, 53, 87, 35},
        {Types::Fighting, Types::Fighting},
    },
    // Hitmonchan
    {
        {50, 105, 79, 76, 35},
        {Types::Fighting, Types::Fighting},
    },
    // Lickitung
    {
        {90, 55, 75, 30, 60},
        {Types::Normal, Types::Normal},
    },
    // Koffing
    {
        {40, 65, 95, 35, 60},
        {Types::Poison, Types::Poison},
    },
    // Weezing
    {
        {65, 90, 120, 60, 85},
        {Types::Poison, Types::Poison},
    },
    // Rhyhorn
    {
        {80, 85, 95, 25, 30},
        {Types::Ground, Types::Rock},
    },
    // Rhydon
    {
        {105, 130, 120, 40, 45},
        {Types::Ground, Types::Rock},
    },
    // Chansey
    {
        {250, 5, 5, 50, 105},
        {Types::Normal, Types::Normal},
    },
    // Tangela
    {
        {65, 55, 115, 60, 100},
        {Types::Grass, Types::Grass},
    },
    // Kangaskhan
    {
        {105, 95, 80, 90, 40},
        {Types::Normal, Types::Normal},
    },
    // Horsea
    {
        {30, 40, 70, 60, 70},
        {Types::Water, Types::Water},
    },
    // Seadra
    {
        {55, 65, 95, 85, 95},
        {Types::Water, Types::Water},
    },
    // Goldeen
    {
        {45, 67, 60, 63, 50},
        {Types::Water, Types::Water},
    },
    // Seaking
    {
        {80, 92, 65, 68, 80},
        {Types::Water, Types::Water},
    },
    // Staryu
    {
        {30, 45, 55, 85, 70},
        {Types::Water, Types::Water},
    },
    // Starmie
    {
        {60, 75, 85, 115, 100},
        {Types::Water, Types::Psychic},
    },
    // MrMime
    {
        {40, 45, 65, 90, 100},
        {Types::Psychic, Types::Psychic},
    },
    // Scyther
    {
        {70, 110, 80, 105, 55},
        {Types::Bug, Types::Flying},
    },
    // Jynx
    {
        {65, 50, 35, 95, 95},
        {Types::Ice, Types::Psychic},
    },
    // Electabuzz
    {
        {65, 83, 57, 105, 85},
        {Types::Electric, Types::Electric},
    },
    // Magmar
    {
        {65, 95, 57, 93, 85},
        {Types::Fire, Types::Fire},
    },
    // Pinsir
    {
        {65, 125, 100, 85, 55},
        {Types::Bug, Types::Bug},
    },
    // Tauros
    {
        {75, 100, 95, 110, 70},
        {Types::Normal, Types::Normal},
    },
    // Magikarp
    {
        {20, 10, 55, 80, 20},
        {Types::Water, Types::Water},
    },
    // Gyarados
    {
        {95, 125, 79, 81, 100},
        {Types::Water, Types::Flying},
    },
    // Lapras
    {
        {130, 85, 80, 60, 95},
        {Types::Water, Types::Ice},
    },
    // Ditto
    {
        {48, 48, 48, 48, 48},
        {Types::Normal, Types::Normal},
    },
    // Eevee
    {
        {55, 55, 50, 55, 65},
        {Types::Normal, Types::Normal},
    },
    // Vaporeon
    {
        {130, 65, 60, 65, 110},
        {Types::Water, Types::Water},
    },
    // Jolteon
    {
        {65, 65, 60, 130, 110},
        {Types::Electric, Types::Electric},
    },
    // Flareon
    {
        {65, 130, 60, 65, 110},
        {Types::Fire, Types::Fire},
    },
    // Porygon
    {
        {65, 60, 70, 40, 75},
        {Types::Normal, Types::Normal},
    },
    // Omanyte
    {
        {35, 40, 100, 35, 90},
        {Types::Rock, Types::Water},
    },
    // Omastar
    {
        {70, 60, 125, 55, 115},
        {Types::Rock, Types::Water},
    },
    // Kabuto
    {
        {30, 80, 90, 55, 45},
        {Types::Rock, Types::Water},
    },
    // Kabutops
    {
        {60, 115, 105, 80, 70},
        {Types::Rock, Types::Water},
    },
    // Aerodactyl
    {
        {80, 105, 65, 130, 60},
        {Types::Rock, Types::Flying},
    },
    // Snorlax
    {
        {160, 110, 65, 30, 65},
        {Types::Normal, Types::Normal},
    },
    // Articuno
    {
        {90, 85, 100, 85, 125},
        {Types::Ice, Types::Flying},
    },
    // Zapdos
    {
        {90, 90, 85, 100, 125},
        {Types::Electric, Types::Flying},
    },
    // Moltres
    {
        {90, 100, 90, 90, 125},
        {Types::Fire, Types::Flying},
    },
    // Dratini
    {
        {41, 64, 45, 50, 50},
        {Types::Dragon, Types::Dragon},
    },
    // Dragonair
    {
        {61, 84, 65, 70, 70},
        {Types::Dragon, Types::Dragon},
    },
    // Dragonite
    {
        {91, 134, 95, 80, 100},
        {Types::Dragon, Types::Flying},
    },
    // Mewtwo
    {
        {106, 110, 90, 130, 154},
        {Types::Psychic, Types::Psychic},
    },
    // Mew
    {
        {100, 100, 100, 100, 100},
        {Types::Psychic, Types::Psychic},
    }};

constexpr auto get_species_data(Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1];
}
constexpr auto get_types(Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1].types;
}

} // namespace Data

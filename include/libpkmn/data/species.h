#pragma once

#include <libpkmn/data/types.h>

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
  std::array<Type, 2> types;
};

constexpr std::array<SpeciesData, 151> SPECIES_DATA{
    // Bulbasaur
    SpeciesData{
        {45, 49, 49, 45, 65},
        {Type::Grass, Type::Poison},
    },
    // Ivysaur
    {
        {60, 62, 63, 60, 80},
        {Type::Grass, Type::Poison},
    },
    // Venusaur
    {
        {80, 82, 83, 80, 100},
        {Type::Grass, Type::Poison},
    },
    // Charmander
    {
        {39, 52, 43, 65, 50},
        {Type::Fire, Type::Fire},
    },
    // Charmeleon
    {
        {58, 64, 58, 80, 65},
        {Type::Fire, Type::Fire},
    },
    // Charizard
    {
        {78, 84, 78, 100, 85},
        {Type::Fire, Type::Flying},
    },
    // Squirtle
    {
        {44, 48, 65, 43, 50},
        {Type::Water, Type::Water},
    },
    // Wartortle
    {
        {59, 63, 80, 58, 65},
        {Type::Water, Type::Water},
    },
    // Blastoise
    {
        {79, 83, 100, 78, 85},
        {Type::Water, Type::Water},
    },
    // Caterpie
    {
        {45, 30, 35, 45, 20},
        {Type::Bug, Type::Bug},
    },
    // Metapod
    {
        {50, 20, 55, 30, 25},
        {Type::Bug, Type::Bug},
    },
    // Butterfree
    {
        {60, 45, 50, 70, 80},
        {Type::Bug, Type::Flying},
    },
    // Weedle
    {
        {40, 35, 30, 50, 20},
        {Type::Bug, Type::Poison},
    },
    // Kakuna
    {
        {45, 25, 50, 35, 25},
        {Type::Bug, Type::Poison},
    },
    // Beedrill
    {
        {65, 80, 40, 75, 45},
        {Type::Bug, Type::Poison},
    },
    // Pidgey
    {
        {40, 45, 40, 56, 35},
        {Type::Normal, Type::Flying},
    },
    // Pidgeotto
    {
        {63, 60, 55, 71, 50},
        {Type::Normal, Type::Flying},
    },
    // Pidgeot
    {
        {83, 80, 75, 91, 70},
        {Type::Normal, Type::Flying},
    },
    // Rattata
    {
        {30, 56, 35, 72, 25},
        {Type::Normal, Type::Normal},
    },
    // Raticate
    {
        {55, 81, 60, 97, 50},
        {Type::Normal, Type::Normal},
    },
    // Spearow
    {
        {40, 60, 30, 70, 31},
        {Type::Normal, Type::Flying},
    },
    // Fearow
    {
        {65, 90, 65, 100, 61},
        {Type::Normal, Type::Flying},
    },
    // Ekans
    {
        {35, 60, 44, 55, 40},
        {Type::Poison, Type::Poison},
    },
    // Arbok
    {
        {60, 85, 69, 80, 65},
        {Type::Poison, Type::Poison},
    },
    // Pikachu
    {
        {35, 55, 30, 90, 50},
        {Type::Electric, Type::Electric},
    },
    // Raichu
    {
        {60, 90, 55, 100, 90},
        {Type::Electric, Type::Electric},
    },
    // Sandshrew
    {
        {50, 75, 85, 40, 30},
        {Type::Ground, Type::Ground},
    },
    // Sandslash
    {
        {75, 100, 110, 65, 55},
        {Type::Ground, Type::Ground},
    },
    // NidoranF
    {
        {55, 47, 52, 41, 40},
        {Type::Poison, Type::Poison},
    },
    // Nidorina
    {
        {70, 62, 67, 56, 55},
        {Type::Poison, Type::Poison},
    },
    // Nidoqueen
    {
        {90, 82, 87, 76, 75},
        {Type::Poison, Type::Ground},
    },
    // NidoranM
    {
        {46, 57, 40, 50, 40},
        {Type::Poison, Type::Poison},
    },
    // Nidorino
    {
        {61, 72, 57, 65, 55},
        {Type::Poison, Type::Poison},
    },
    // Nidoking
    {
        {81, 92, 77, 85, 75},
        {Type::Poison, Type::Ground},
    },
    // Clefairy
    {
        {70, 45, 48, 35, 60},
        {Type::Normal, Type::Normal},
    },
    // Clefable
    {
        {95, 70, 73, 60, 85},
        {Type::Normal, Type::Normal},
    },
    // Vulpix
    {
        {38, 41, 40, 65, 65},
        {Type::Fire, Type::Fire},
    },
    // Ninetales
    {
        {73, 76, 75, 100, 100},
        {Type::Fire, Type::Fire},
    },
    // Jigglypuff
    {
        {115, 45, 20, 20, 25},
        {Type::Normal, Type::Normal},
    },
    // Wigglytuff
    {
        {140, 70, 45, 45, 50},
        {Type::Normal, Type::Normal},
    },
    // Zubat
    {
        {40, 45, 35, 55, 40},
        {Type::Poison, Type::Flying},
    },
    // Golbat
    {
        {75, 80, 70, 90, 75},
        {Type::Poison, Type::Flying},
    },
    // Oddish
    {
        {45, 50, 55, 30, 75},
        {Type::Grass, Type::Poison},
    },
    // Gloom
    {
        {60, 65, 70, 40, 85},
        {Type::Grass, Type::Poison},
    },
    // Vileplume
    {
        {75, 80, 85, 50, 100},
        {Type::Grass, Type::Poison},
    },
    // Paras
    {
        {35, 70, 55, 25, 55},
        {Type::Bug, Type::Grass},
    },
    // Parasect
    {
        {60, 95, 80, 30, 80},
        {Type::Bug, Type::Grass},
    },
    // Venonat
    {
        {60, 55, 50, 45, 40},
        {Type::Bug, Type::Poison},
    },
    // Venomoth
    {
        {70, 65, 60, 90, 90},
        {Type::Bug, Type::Poison},
    },
    // Diglett
    {
        {10, 55, 25, 95, 45},
        {Type::Ground, Type::Ground},
    },
    // Dugtrio
    {
        {35, 80, 50, 120, 70},
        {Type::Ground, Type::Ground},
    },
    // Meowth
    {
        {40, 45, 35, 90, 40},
        {Type::Normal, Type::Normal},
    },
    // Persian
    {
        {65, 70, 60, 115, 65},
        {Type::Normal, Type::Normal},
    },
    // Psyduck
    {
        {50, 52, 48, 55, 50},
        {Type::Water, Type::Water},
    },
    // Golduck
    {
        {80, 82, 78, 85, 80},
        {Type::Water, Type::Water},
    },
    // Mankey
    {
        {40, 80, 35, 70, 35},
        {Type::Fighting, Type::Fighting},
    },
    // Primeape
    {
        {65, 105, 60, 95, 60},
        {Type::Fighting, Type::Fighting},
    },
    // Growlithe
    {
        {55, 70, 45, 60, 50},
        {Type::Fire, Type::Fire},
    },
    // Arcanine
    {
        {90, 110, 80, 95, 80},
        {Type::Fire, Type::Fire},
    },
    // Poliwag
    {
        {40, 50, 40, 90, 40},
        {Type::Water, Type::Water},
    },
    // Poliwhirl
    {
        {65, 65, 65, 90, 50},
        {Type::Water, Type::Water},
    },
    // Poliwrath
    {
        {90, 85, 95, 70, 70},
        {Type::Water, Type::Fighting},
    },
    // Abra
    {
        {25, 20, 15, 90, 105},
        {Type::Psychic, Type::Psychic},
    },
    // Kadabra
    {
        {40, 35, 30, 105, 120},
        {Type::Psychic, Type::Psychic},
    },
    // Alakazam
    {
        {55, 50, 45, 120, 135},
        {Type::Psychic, Type::Psychic},
    },
    // Machop
    {
        {70, 80, 50, 35, 35},
        {Type::Fighting, Type::Fighting},
    },
    // Machoke
    {
        {80, 100, 70, 45, 50},
        {Type::Fighting, Type::Fighting},
    },
    // Machamp
    {
        {90, 130, 80, 55, 65},
        {Type::Fighting, Type::Fighting},
    },
    // Bellsprout
    {
        {50, 75, 35, 40, 70},
        {Type::Grass, Type::Poison},
    },
    // Weepinbell
    {
        {65, 90, 50, 55, 85},
        {Type::Grass, Type::Poison},
    },
    // Victreebel
    {
        {80, 105, 65, 70, 100},
        {Type::Grass, Type::Poison},
    },
    // Tentacool
    {
        {40, 40, 35, 70, 100},
        {Type::Water, Type::Poison},
    },
    // Tentacruel
    {
        {80, 70, 65, 100, 120},
        {Type::Water, Type::Poison},
    },
    // Geodude
    {
        {40, 80, 100, 20, 30},
        {Type::Rock, Type::Ground},
    },
    // Graveler
    {
        {55, 95, 115, 35, 45},
        {Type::Rock, Type::Ground},
    },
    // Golem
    {
        {80, 110, 130, 45, 55},
        {Type::Rock, Type::Ground},
    },
    // Ponyta
    {
        {50, 85, 55, 90, 65},
        {Type::Fire, Type::Fire},
    },
    // Rapidash
    {
        {65, 100, 70, 105, 80},
        {Type::Fire, Type::Fire},
    },
    // Slowpoke
    {
        {90, 65, 65, 15, 40},
        {Type::Water, Type::Psychic},
    },
    // Slowbro
    {
        {95, 75, 110, 30, 80},
        {Type::Water, Type::Psychic},
    },
    // Magnemite
    {
        {25, 35, 70, 45, 95},
        {Type::Electric, Type::Electric},
    },
    // Magneton
    {
        {50, 60, 95, 70, 120},
        {Type::Electric, Type::Electric},
    },
    // Farfetchd
    {
        {52, 65, 55, 60, 58},
        {Type::Normal, Type::Flying},
    },
    // Doduo
    {
        {35, 85, 45, 75, 35},
        {Type::Normal, Type::Flying},
    },
    // Dodrio
    {
        {60, 110, 70, 100, 60},
        {Type::Normal, Type::Flying},
    },
    // Seel
    {
        {65, 45, 55, 45, 70},
        {Type::Water, Type::Water},
    },
    // Dewgong
    {
        {90, 70, 80, 70, 95},
        {Type::Water, Type::Ice},
    },
    // Grimer
    {
        {80, 80, 50, 25, 40},
        {Type::Poison, Type::Poison},
    },
    // Muk
    {
        {105, 105, 75, 50, 65},
        {Type::Poison, Type::Poison},
    },
    // Shellder
    {
        {30, 65, 100, 40, 45},
        {Type::Water, Type::Water},
    },
    // Cloyster
    {
        {50, 95, 180, 70, 85},
        {Type::Water, Type::Ice},
    },
    // Gastly
    {
        {30, 35, 30, 80, 100},
        {Type::Ghost, Type::Poison},
    },
    // Haunter
    {
        {45, 50, 45, 95, 115},
        {Type::Ghost, Type::Poison},
    },
    // Gengar
    {
        {60, 65, 60, 110, 130},
        {Type::Ghost, Type::Poison},
    },
    // Onix
    {
        {35, 45, 160, 70, 30},
        {Type::Rock, Type::Ground},
    },
    // Drowzee
    {
        {60, 48, 45, 42, 90},
        {Type::Psychic, Type::Psychic},
    },
    // Hypno
    {
        {85, 73, 70, 67, 115},
        {Type::Psychic, Type::Psychic},
    },
    // Krabby
    {
        {30, 105, 90, 50, 25},
        {Type::Water, Type::Water},
    },
    // Kingler
    {
        {55, 130, 115, 75, 50},
        {Type::Water, Type::Water},
    },
    // Voltorb
    {
        {40, 30, 50, 100, 55},
        {Type::Electric, Type::Electric},
    },
    // Electrode
    {
        {60, 50, 70, 140, 80},
        {Type::Electric, Type::Electric},
    },
    // Exeggcute
    {
        {60, 40, 80, 40, 60},
        {Type::Grass, Type::Psychic},
    },
    // Exeggutor
    {
        {95, 95, 85, 55, 125},
        {Type::Grass, Type::Psychic},
    },
    // Cubone
    {
        {50, 50, 95, 35, 40},
        {Type::Ground, Type::Ground},
    },
    // Marowak
    {
        {60, 80, 110, 45, 50},
        {Type::Ground, Type::Ground},
    },
    // Hitmonlee
    {
        {50, 120, 53, 87, 35},
        {Type::Fighting, Type::Fighting},
    },
    // Hitmonchan
    {
        {50, 105, 79, 76, 35},
        {Type::Fighting, Type::Fighting},
    },
    // Lickitung
    {
        {90, 55, 75, 30, 60},
        {Type::Normal, Type::Normal},
    },
    // Koffing
    {
        {40, 65, 95, 35, 60},
        {Type::Poison, Type::Poison},
    },
    // Weezing
    {
        {65, 90, 120, 60, 85},
        {Type::Poison, Type::Poison},
    },
    // Rhyhorn
    {
        {80, 85, 95, 25, 30},
        {Type::Ground, Type::Rock},
    },
    // Rhydon
    {
        {105, 130, 120, 40, 45},
        {Type::Ground, Type::Rock},
    },
    // Chansey
    {
        {250, 5, 5, 50, 105},
        {Type::Normal, Type::Normal},
    },
    // Tangela
    {
        {65, 55, 115, 60, 100},
        {Type::Grass, Type::Grass},
    },
    // Kangaskhan
    {
        {105, 95, 80, 90, 40},
        {Type::Normal, Type::Normal},
    },
    // Horsea
    {
        {30, 40, 70, 60, 70},
        {Type::Water, Type::Water},
    },
    // Seadra
    {
        {55, 65, 95, 85, 95},
        {Type::Water, Type::Water},
    },
    // Goldeen
    {
        {45, 67, 60, 63, 50},
        {Type::Water, Type::Water},
    },
    // Seaking
    {
        {80, 92, 65, 68, 80},
        {Type::Water, Type::Water},
    },
    // Staryu
    {
        {30, 45, 55, 85, 70},
        {Type::Water, Type::Water},
    },
    // Starmie
    {
        {60, 75, 85, 115, 100},
        {Type::Water, Type::Psychic},
    },
    // MrMime
    {
        {40, 45, 65, 90, 100},
        {Type::Psychic, Type::Psychic},
    },
    // Scyther
    {
        {70, 110, 80, 105, 55},
        {Type::Bug, Type::Flying},
    },
    // Jynx
    {
        {65, 50, 35, 95, 95},
        {Type::Ice, Type::Psychic},
    },
    // Electabuzz
    {
        {65, 83, 57, 105, 85},
        {Type::Electric, Type::Electric},
    },
    // Magmar
    {
        {65, 95, 57, 93, 85},
        {Type::Fire, Type::Fire},
    },
    // Pinsir
    {
        {65, 125, 100, 85, 55},
        {Type::Bug, Type::Bug},
    },
    // Tauros
    {
        {75, 100, 95, 110, 70},
        {Type::Normal, Type::Normal},
    },
    // Magikarp
    {
        {20, 10, 55, 80, 20},
        {Type::Water, Type::Water},
    },
    // Gyarados
    {
        {95, 125, 79, 81, 100},
        {Type::Water, Type::Flying},
    },
    // Lapras
    {
        {130, 85, 80, 60, 95},
        {Type::Water, Type::Ice},
    },
    // Ditto
    {
        {48, 48, 48, 48, 48},
        {Type::Normal, Type::Normal},
    },
    // Eevee
    {
        {55, 55, 50, 55, 65},
        {Type::Normal, Type::Normal},
    },
    // Vaporeon
    {
        {130, 65, 60, 65, 110},
        {Type::Water, Type::Water},
    },
    // Jolteon
    {
        {65, 65, 60, 130, 110},
        {Type::Electric, Type::Electric},
    },
    // Flareon
    {
        {65, 130, 60, 65, 110},
        {Type::Fire, Type::Fire},
    },
    // Porygon
    {
        {65, 60, 70, 40, 75},
        {Type::Normal, Type::Normal},
    },
    // Omanyte
    {
        {35, 40, 100, 35, 90},
        {Type::Rock, Type::Water},
    },
    // Omastar
    {
        {70, 60, 125, 55, 115},
        {Type::Rock, Type::Water},
    },
    // Kabuto
    {
        {30, 80, 90, 55, 45},
        {Type::Rock, Type::Water},
    },
    // Kabutops
    {
        {60, 115, 105, 80, 70},
        {Type::Rock, Type::Water},
    },
    // Aerodactyl
    {
        {80, 105, 65, 130, 60},
        {Type::Rock, Type::Flying},
    },
    // Snorlax
    {
        {160, 110, 65, 30, 65},
        {Type::Normal, Type::Normal},
    },
    // Articuno
    {
        {90, 85, 100, 85, 125},
        {Type::Ice, Type::Flying},
    },
    // Zapdos
    {
        {90, 90, 85, 100, 125},
        {Type::Electric, Type::Flying},
    },
    // Moltres
    {
        {90, 100, 90, 90, 125},
        {Type::Fire, Type::Flying},
    },
    // Dratini
    {
        {41, 64, 45, 50, 50},
        {Type::Dragon, Type::Dragon},
    },
    // Dragonair
    {
        {61, 84, 65, 70, 70},
        {Type::Dragon, Type::Dragon},
    },
    // Dragonite
    {
        {91, 134, 95, 80, 100},
        {Type::Dragon, Type::Flying},
    },
    // Mewtwo
    {
        {106, 110, 90, 130, 154},
        {Type::Psychic, Type::Psychic},
    },
    // Mew
    {
        {100, 100, 100, 100, 100},
        {Type::Psychic, Type::Psychic},
    }};

constexpr auto get_species_data(Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1];
}
constexpr auto get_types(Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1].types;
}

} // namespace Data

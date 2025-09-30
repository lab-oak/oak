#pragma once

#include <data/moves.h>
#include <data/species.h>
#include <data/types.h>

namespace RandomBattlesData {

using Data::Species;
using Data::Types;

static constexpr std::array<Species, 146> pokemonPool{
    Species::Bulbasaur,  Species::Ivysaur,    Species::Venusaur,
    Species::Charmander, Species::Charmeleon, Species::Charizard,
    Species::Squirtle,   Species::Wartortle,  Species::Blastoise,
    Species::Butterfree, Species::Beedrill,   Species::Pidgey,
    Species::Pidgeotto,  Species::Pidgeot,    Species::Rattata,
    Species::Raticate,   Species::Spearow,    Species::Fearow,
    Species::Ekans,      Species::Arbok,      Species::Pikachu,
    Species::Raichu,     Species::Sandshrew,  Species::Sandslash,
    Species::NidoranF,   Species::Nidorina,   Species::Nidoqueen,
    Species::NidoranM,   Species::Nidorino,   Species::Nidoking,
    Species::Clefairy,   Species::Clefable,   Species::Vulpix,
    Species::Ninetales,  Species::Jigglypuff, Species::Wigglytuff,
    Species::Zubat,      Species::Golbat,     Species::Oddish,
    Species::Gloom,      Species::Vileplume,  Species::Paras,
    Species::Parasect,   Species::Venonat,    Species::Venomoth,
    Species::Diglett,    Species::Dugtrio,    Species::Meowth,
    Species::Persian,    Species::Psyduck,    Species::Golduck,
    Species::Mankey,     Species::Primeape,   Species::Growlithe,
    Species::Arcanine,   Species::Poliwag,    Species::Poliwhirl,
    Species::Poliwrath,  Species::Abra,       Species::Kadabra,
    Species::Alakazam,   Species::Machop,     Species::Machoke,
    Species::Machamp,    Species::Bellsprout, Species::Weepinbell,
    Species::Victreebel, Species::Tentacool,  Species::Tentacruel,
    Species::Geodude,    Species::Graveler,   Species::Golem,
    Species::Ponyta,     Species::Rapidash,   Species::Slowpoke,
    Species::Slowbro,    Species::Magnemite,  Species::Magneton,
    Species::Farfetchd,  Species::Doduo,      Species::Dodrio,
    Species::Seel,       Species::Dewgong,    Species::Grimer,
    Species::Muk,        Species::Shellder,   Species::Cloyster,
    Species::Gastly,     Species::Haunter,    Species::Gengar,
    Species::Onix,       Species::Drowzee,    Species::Hypno,
    Species::Krabby,     Species::Kingler,    Species::Voltorb,
    Species::Electrode,  Species::Exeggcute,  Species::Exeggutor,
    Species::Cubone,     Species::Marowak,    Species::Hitmonlee,
    Species::Hitmonchan, Species::Lickitung,  Species::Koffing,
    Species::Weezing,    Species::Rhyhorn,    Species::Rhydon,
    Species::Chansey,    Species::Tangela,    Species::Kangaskhan,
    Species::Horsea,     Species::Seadra,     Species::Goldeen,
    Species::Seaking,    Species::Staryu,     Species::Starmie,
    Species::MrMime,     Species::Scyther,    Species::Jynx,
    Species::Electabuzz, Species::Magmar,     Species::Pinsir,
    Species::Tauros,     Species::Gyarados,   Species::Lapras,
    Species::Ditto,      Species::Eevee,      Species::Vaporeon,
    Species::Jolteon,    Species::Flareon,    Species::Porygon,
    Species::Omanyte,    Species::Omastar,    Species::Kabuto,
    Species::Kabutops,   Species::Aerodactyl, Species::Snorlax,
    Species::Articuno,   Species::Zapdos,     Species::Moltres,
    Species::Dratini,    Species::Dragonair,  Species::Dragonite,
    Species::Mewtwo,     Species::Mew};

constexpr bool isLevel100(Species species) noexcept {
  return (species == Species::Ditto) || (species == Species::Zubat);
}

// {Electric: 0, Psychic: 0, Water: 0, Ice: 0, Ground: 0, Fire: 0};
constexpr std::array<Data::Types, 6> importantTypes{
    Types::Electric, Types::Psychic, Types::Water,
    Types::Ice,      Types::Ground,  Types::Fire};

// template <Data::Types types, int foo, typename T>
consteval std::array<uint8_t, 6> getImportantWeaknesses(const Species species) {
  const auto types =
      Data::SPECIES_DATA[static_cast<uint8_t>(species) - 1].types;
  return {
      static_cast<uint8_t>(Data::get_effectiveness(Types::Electric, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Electric, types[1])) >
          4,
      static_cast<uint8_t>(Data::get_effectiveness(Types::Psychic, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Psychic, types[1])) >
          4,
      static_cast<uint8_t>(Data::get_effectiveness(Types::Water, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Water, types[1])) >
          4,
      static_cast<uint8_t>(Data::get_effectiveness(Types::Ice, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Ice, types[1])) >
          4,
      static_cast<uint8_t>(Data::get_effectiveness(Types::Ground, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Ground, types[1])) >
          4,
      static_cast<uint8_t>(Data::get_effectiveness(Types::Fire, types[0])) *
              static_cast<uint8_t>(
                  Data::get_effectiveness(Types::Fire, types[1])) >
          4,
  };
}

consteval std::array<std::array<uint8_t, 6>, 152> IMPORTANT_WEAKNESSES() {
  std::array<std::array<uint8_t, 6>, 152> result{};
  for (int i = 0; i < 152; ++i) {
    result[i] = getImportantWeaknesses(static_cast<Species>(i));
  }
  return result;
}

static_assert(getImportantWeaknesses(Species::Parasect)[0] == false &&
              getImportantWeaknesses(Species::Parasect)[1] == false &&
              getImportantWeaknesses(Species::Parasect)[2] == false &&
              getImportantWeaknesses(Species::Parasect)[3] == true &&
              getImportantWeaknesses(Species::Parasect)[4] == false &&
              getImportantWeaknesses(Species::Parasect)[5] == true);

}; // namespace RandomBattlesData

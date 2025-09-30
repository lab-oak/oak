const set_json = require("../extern/pokemon-showdown/data/random-battles/gen1/data.json");

// only used to get all_moves_precomputed and feed that into GPT
function get_all_moves_set(): set<string> {
    let all_moves: Set<string> = new Set();

    for (let species in set_json) {
        const data = set_json[species];
        for (const thing of Object.values(data)) {
            if (Number.isInteger(thing)) {
                continue;
            }
            for (const move of thing as string[]) {
                all_moves.add(move);
            }
        }
    }
    return all_moves;
}

let max_moves: number = 0;
let max_exclusive_moves: number = 0;
let max_essential_moves: number = 0;
let max_combo_moves: number = 0;

let n_important_types: number = 6;

function get_all_max_counts(): void {
    for (let species in set_json) {
        const data = set_json[species];
        max_moves = Math.max(max_moves, data.moves?.length || 0);
        max_exclusive_moves = Math.max(max_exclusive_moves, data.exclusiveMoves?.length || 0);
        max_essential_moves = Math.max(max_essential_moves, data.essentialMoves?.length || 0);
        max_combo_moves = Math.max(max_combo_moves, data.comboMoves?.length || 0);
    }
}

get_all_max_counts();

function fixName(move: string): string {
    const all_moves_precomputed = ["bodyslam",
        "razorleaf",
        "sleeppowder",
        "swordsdance",
        "hyperbeam",
        "counter",
        "seismictoss",
        "slash",
        "fireblast",
        "submission",
        "earthquake",
        "blizzard",
        "hydropump",
        "surf",
        "rest",
        "psychic",
        "stunspore",
        "doubleedge",
        "megadrain",
        "substitute",
        "twineedle",
        "agility",
        "quickattack",
        "skyattack",
        "mirrormove",
        "sandattack",
        "reflect",
        "superfang",
        "thunderbolt",
        "drillpeck",
        "leer",
        "mimic",
        "glare",
        "rockslide",
        "thunderwave",
        "thunder",
        "doublekick",
        "bubblebeam",
        "sing",
        "confuseray",
        "flamethrower",
        "wingattack",
        "spore",
        "amnesia",
        "lowkick",
        "megakick",
        "hypnosis",
        "recover",
        "barrier",
        "explosion",
        "stomp",
        "sludge",
        "nightshade",
        "crabhammer",
        "takedown",
        "highjumpkick",
        "meditate",
        "rollingkick",
        "icebeam",
        "softboiled",
        "growth",
        "smokescreen",
        "lovelykiss",
        "transform",
        "tailwhip",
        "acidarmor",
        "pinmissile",
        "triattack",
        "selfdestruct"];

    const all_moves_fixed = [
        "Moves::BodySlam",
        "Moves::RazorLeaf",
        "Moves::SleepPowder",
        "Moves::SwordsDance",
        "Moves::HyperBeam",
        "Moves::Counter",
        "Moves::SeismicToss",
        "Moves::Slash",
        "Moves::FireBlast",
        "Moves::Submission",
        "Moves::Earthquake",
        "Moves::Blizzard",
        "Moves::HydroPump",
        "Moves::Surf",
        "Moves::Rest",
        "Moves::Psychic",
        "Moves::StunSpore",
        "Moves::DoubleEdge",
        "Moves::MegaDrain",
        "Moves::Substitute",
        "Moves::Twineedle",
        "Moves::Agility",
        "Moves::QuickAttack",
        "Moves::SkyAttack",
        "Moves::MirrorMove",
        "Moves::SandAttack",
        "Moves::Reflect",
        "Moves::SuperFang",
        "Moves::Thunderbolt",
        "Moves::DrillPeck",
        "Moves::Leer",
        "Moves::Mimic",
        "Moves::Glare",
        "Moves::RockSlide",
        "Moves::ThunderWave",
        "Moves::Thunder",
        "Moves::DoubleKick",
        "Moves::BubbleBeam",
        "Moves::Sing",
        "Moves::ConfuseRay",
        "Moves::Flamethrower",
        "Moves::WingAttack",
        "Moves::Spore",
        "Moves::Amnesia",
        "Moves::LowKick",
        "Moves::MegaKick",
        "Moves::Hypnosis",
        "Moves::Recover",
        "Moves::Barrier",
        "Moves::Explosion",
        "Moves::Stomp",
        "Moves::Sludge",
        "Moves::NightShade",
        "Moves::Crabhammer",
        "Moves::TakeDown",
        "Moves::HighJumpKick",
        "Moves::Meditate",
        "Moves::RollingKick",
        "Moves::IceBeam",
        "Moves::SoftBoiled",
        "Moves::Growth",
        "Moves::Smokescreen",
        "Moves::LovelyKiss",
        "Moves::Transform",
        "Moves::TailWhip",
        "Moves::AcidArmor",
        "Moves::PinMissile",
        "Moves::TriAttack",
        "Moves::SelfDestruct"
    ];

    // intended behaviour, for when we go out of bounds
    if (move === undefined) {
        return "Moves::None";
    }

    const index = all_moves_precomputed.indexOf(move);
    if (index === -1) {
        console.error("bad move string");
    }
    return all_moves_fixed[index];
}

function print_set_data_as_initializer(libpkmn_species: string): boolean {
    const species = libpkmn_species.toLowerCase();
    const data: any = set_json[species];
    if (data === undefined) {
        console.log("RandomSetEntry{},");
        return false;
    }
    const level: Number = data.level || 100;
    const moves: string[] = data.moves || [];
    const exclusiveMoves: string[] = data.exclusiveMoves || [];
    const essentialMoves: string[] = data.essentialMoves || [];
    const comboMoves: string[] = data.comboMoves || [];

    const s =
        `RandomSetEntry{ ${level},${moves.length},${exclusiveMoves.length},${essentialMoves.length},${comboMoves.length},
    {${Array.from({ length: max_moves }, (_, x) => fixName(moves[x]))}},
    {${Array.from({ length: max_exclusive_moves }, (_, x) => fixName(exclusiveMoves[x]))}},
    {${Array.from({ length: max_essential_moves }, (_, x) => fixName(essentialMoves[x]))}},
    {${Array.from({ length: max_combo_moves }, (_, x) => fixName(comboMoves[x]))}},
    getImportantWeaknesses(Species::${libpkmn_species}), Data::get_types(Species::${libpkmn_species})},`;

    console.log(s);
    return true;
}

const libpkmn_species: string[] = [
    "None",
    "Bulbasaur",
    "Ivysaur",
    "Venusaur",
    "Charmander",
    "Charmeleon",
    "Charizard",
    "Squirtle",
    "Wartortle",
    "Blastoise",
    "Caterpie",
    "Metapod",
    "Butterfree",
    "Weedle",
    "Kakuna",
    "Beedrill",
    "Pidgey",
    "Pidgeotto",
    "Pidgeot",
    "Rattata",
    "Raticate",
    "Spearow",
    "Fearow",
    "Ekans",
    "Arbok",
    "Pikachu",
    "Raichu",
    "Sandshrew",
    "Sandslash",
    "NidoranF",
    "Nidorina",
    "Nidoqueen",
    "NidoranM",
    "Nidorino",
    "Nidoking",
    "Clefairy",
    "Clefable",
    "Vulpix",
    "Ninetales",
    "Jigglypuff",
    "Wigglytuff",
    "Zubat",
    "Golbat",
    "Oddish",
    "Gloom",
    "Vileplume",
    "Paras",
    "Parasect",
    "Venonat",
    "Venomoth",
    "Diglett",
    "Dugtrio",
    "Meowth",
    "Persian",
    "Psyduck",
    "Golduck",
    "Mankey",
    "Primeape",
    "Growlithe",
    "Arcanine",
    "Poliwag",
    "Poliwhirl",
    "Poliwrath",
    "Abra",
    "Kadabra",
    "Alakazam",
    "Machop",
    "Machoke",
    "Machamp",
    "Bellsprout",
    "Weepinbell",
    "Victreebel",
    "Tentacool",
    "Tentacruel",
    "Geodude",
    "Graveler",
    "Golem",
    "Ponyta",
    "Rapidash",
    "Slowpoke",
    "Slowbro",
    "Magnemite",
    "Magneton",
    "Farfetchd",
    "Doduo",
    "Dodrio",
    "Seel",
    "Dewgong",
    "Grimer",
    "Muk",
    "Shellder",
    "Cloyster",
    "Gastly",
    "Haunter",
    "Gengar",
    "Onix",
    "Drowzee",
    "Hypno",
    "Krabby",
    "Kingler",
    "Voltorb",
    "Electrode",
    "Exeggcute",
    "Exeggutor",
    "Cubone",
    "Marowak",
    "Hitmonlee",
    "Hitmonchan",
    "Lickitung",
    "Koffing",
    "Weezing",
    "Rhyhorn",
    "Rhydon",
    "Chansey",
    "Tangela",
    "Kangaskhan",
    "Horsea",
    "Seadra",
    "Goldeen",
    "Seaking",
    "Staryu",
    "Starmie",
    "MrMime",
    "Scyther",
    "Jynx",
    "Electabuzz",
    "Magmar",
    "Pinsir",
    "Tauros",
    "Magikarp",
    "Gyarados",
    "Lapras",
    "Ditto",
    "Eevee",
    "Vaporeon",
    "Jolteon",
    "Flareon",
    "Porygon",
    "Omanyte",
    "Omastar",
    "Kabuto",
    "Kabutops",
    "Aerodactyl",
    "Snorlax",
    "Articuno",
    "Zapdos",
    "Moltres",
    "Dratini",
    "Dragonair",
    "Dragonite",
    "Mewtwo",
    "Mew"];

const header_header: string =
    `
#pragma once

#include <array>

#include "data.h"

namespace RandomBattlesData {

using Data::Species;
using Data::Moves;
using Data::Types;

struct RandomSetEntry {
  static constexpr uint8_t max_moves{${max_moves}};
  static constexpr uint8_t max_exclusive_moves{${max_exclusive_moves}};
  static constexpr uint8_t max_essential_moves{${max_essential_moves}};
  static constexpr uint8_t max_combo_moves{${max_combo_moves}};

  uint8_t level;
  uint8_t n_moves;
  uint8_t n_exclusive_moves;
  uint8_t n_essential_moves;
  uint8_t n_combo_moves;

  std::array<Moves, max_moves> moves;
  std::array<Moves, max_exclusive_moves> exclusive_moves;
  std::array<Moves, max_essential_moves> essential_moves;
  std::array<Moves, max_combo_moves> combo_moves;

  std::array<uint8_t, ${n_important_types}> weaknesses{};
  std::array<Types, 2> types{};
};

constexpr std::array<RandomSetEntry, 152> RANDOM_SET_DATA
{
`;

// cd cheats && bun ./convert.ts > random-set-data.h

function main() {

    console.log(header_header);

    for (let species of libpkmn_species) {
        if (!print_set_data_as_initializer(species)) {
            console.log("//", species.toLowerCase(), " not found in data.json");
        }
    }

    console.log("}; // RANDOM_SET_DATA \n}; // namespace RandomSetData");

}

main();
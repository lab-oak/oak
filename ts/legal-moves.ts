// GPT script for creating legal-moves.h

import { Dex } from '@pkmn/dex';
import { Generations } from '@pkmn/data';
import * as fs from 'fs';
import assert from 'assert';

const gens = new Generations(Dex);
const gen1 = gens.get(1);

const moveEnumNames = [
  "None",
  "Pound",
  "KarateChop",
  "DoubleSlap",
  "CometPunch",
  "MegaPunch",
  "PayDay",
  "FirePunch",
  "IcePunch",
  "ThunderPunch",
  "Scratch",
  "ViseGrip",
  "Guillotine",
  "RazorWind",
  "SwordsDance",
  "Cut",
  "Gust",
  "WingAttack",
  "Whirlwind",
  "Fly",
  "Bind",
  "Slam",
  "VineWhip",
  "Stomp",
  "DoubleKick",
  "MegaKick",
  "JumpKick",
  "RollingKick",
  "SandAttack",
  "Headbutt",
  "HornAttack",
  "FuryAttack",
  "HornDrill",
  "Tackle",
  "BodySlam",
  "Wrap",
  "TakeDown",
  "Thrash",
  "DoubleEdge",
  "TailWhip",
  "PoisonSting",
  "Twineedle",
  "PinMissile",
  "Leer",
  "Bite",
  "Growl",
  "Roar",
  "Sing",
  "Supersonic",
  "SonicBoom",
  "Disable",
  "Acid",
  "Ember",
  "Flamethrower",
  "Mist",
  "WaterGun",
  "HydroPump",
  "Surf",
  "IceBeam",
  "Blizzard",
  "Psybeam",
  "BubbleBeam",
  "AuroraBeam",
  "HyperBeam",
  "Peck",
  "DrillPeck",
  "Submission",
  "LowKick",
  "Counter",
  "SeismicToss",
  "Strength",
  "Absorb",
  "MegaDrain",
  "LeechSeed",
  "Growth",
  "RazorLeaf",
  "SolarBeam",
  "PoisonPowder",
  "StunSpore",
  "SleepPowder",
  "PetalDance",
  "StringShot",
  "DragonRage",
  "FireSpin",
  "ThunderShock",
  "Thunderbolt",
  "ThunderWave",
  "Thunder",
  "RockThrow",
  "Earthquake",
  "Fissure",
  "Dig",
  "Toxic",
  "Confusion",
  "Psychic",
  "Hypnosis",
  "Meditate",
  "Agility",
  "QuickAttack",
  "Rage",
  "Teleport",
  "NightShade",
  "Mimic",
  "Screech",
  "DoubleTeam",
  "Recover",
  "Harden",
  "Minimize",
  "Smokescreen",
  "ConfuseRay",
  "Withdraw",
  "DefenseCurl",
  "Barrier",
  "LightScreen",
  "Haze",
  "Reflect",
  "FocusEnergy",
  "Bide",
  "Metronome",
  "MirrorMove",
  "SelfDestruct",
  "EggBomb",
  "Lick",
  "Smog",
  "Sludge",
  "BoneClub",
  "FireBlast",
  "Waterfall",
  "Clamp",
  "Swift",
  "SkullBash",
  "SpikeCannon",
  "Constrict",
  "Amnesia",
  "Kinesis",
  "SoftBoiled",
  "HighJumpKick",
  "Glare",
  "DreamEater",
  "PoisonGas",
  "Barrage",
  "LeechLife",
  "LovelyKiss",
  "SkyAttack",
  "Transform",
  "Bubble",
  "DizzyPunch",
  "Spore",
  "Flash",
  "Psywave",
  "Splash",
  "AcidArmor",
  "Crabhammer",
  "Explosion",
  "FurySwipes",
  "Bonemerang",
  "Rest",
  "RockSlide",
  "HyperFang",
  "Sharpen",
  "Conversion",
  "TriAttack",
  "SuperFang",
  "Slash",
  "Substitute",
  "Struggle",
];

// Build map: move name lowercase -> move enum index
const moveNameToIndex = new Map<string, number>();
for (let i = 0; i < moveEnumNames.length; i++) {
  moveNameToIndex.set(moveEnumNames[i].toLowerCase(), i);
}

// Get Gen 1 Pokémon with dex numbers 1 to 151
const gen1Mons = Array.from(Dex.species.all())
  .filter(mon => mon.gen === 1 && mon.num >= 1 && mon.num <= 151)
  .sort((a, b) => a.num - b.num);

// Create a 2D boolean array [pokemon][move]
const numPokemon = 151;
const numMoves = moveEnumNames.length;
assert(numMoves == 166);

async function generate() {
  const learnsets = gen1.learnsets;

  // Initialize array with false
  const legalMovesArray: boolean[][] = [];
  for (let i = 0; i < numPokemon; i++) {
    legalMovesArray[i] = new Array(numMoves).fill(false);
  }

  for (const mon of gen1Mons) {    
    const speciesIndex = mon.num - 1;
    const id = mon.id;
    const pokemonLearnset = await learnsets.get(id);

    if (!pokemonLearnset?.learnset) continue;

    for (const move in pokemonLearnset.learnset) {
      // Check if move is in our enum list
      const moveIndex = moveNameToIndex.get(move.toLowerCase());
      if (moveIndex === undefined) continue;

      // Check if move is legal in Gen 1
      const sources = pokemonLearnset.learnset[move];
      if (sources.some(src => src.startsWith('1'))) {
        legalMovesArray[speciesIndex][moveIndex] = true;
      }
    }

    if (mon.prevo!) {
      const prevoIndex = Dex.species.get(mon.prevo!)?.num - 1;
      if (prevoIndex > 151) {
        continue;
      }
      console.log(speciesIndex, prevoIndex);
      for (let i = 0; i < numMoves; ++i) {
        legalMovesArray[speciesIndex][i] ||= legalMovesArray[prevoIndex][i];
      }
    }
  }

  // Generate C++ header content
  let header = `#pragma once

#include <array>

namespace Data {
constexpr std::array<std::array<bool, 166>, 151> legal_moves = {\n`;

  for (let i = 0; i < numPokemon; i++) {
    header += "  std::array<bool, 166>{ ";
    for (let j = 0; j < numMoves; j++) {
      header += legalMovesArray[i][j] ? "true" : "false";
      if (j !== numMoves - 1) header += ", ";
    }
    header += " }";
    if (i !== numPokemon - 1) header += ",\n";
  }

  header += "\n};\n";
  header += `
  } // namespace Data
  `

  // Optionally write to file
  fs.writeFileSync("legal-moves.h", header);
  console.log("Header file legal-moves.h generated!");
}

generate();

#pragma once

#include <data/types.h>

#include <array>
#include <cstdint>

namespace Data {

enum class Move : std::underlying_type_t<std::byte> {
  None,
  Pound,
  KarateChop,
  DoubleSlap,
  CometPunch,
  MegaPunch,
  PayDay,
  FirePunch,
  IcePunch,
  ThunderPunch,
  Scratch,
  ViseGrip,
  Guillotine,
  RazorWind,
  SwordsDance,
  Cut,
  Gust,
  WingAttack,
  Whirlwind,
  Fly,
  Bind,
  Slam,
  VineWhip,
  Stomp,
  DoubleKick,
  MegaKick,
  JumpKick,
  RollingKick,
  SandAttack,
  Headbutt,
  HornAttack,
  FuryAttack,
  HornDrill,
  Tackle,
  BodySlam,
  Wrap,
  TakeDown,
  Thrash,
  DoubleEdge,
  TailWhip,
  PoisonSting,
  Twineedle,
  PinMissile,
  Leer,
  Bite,
  Growl,
  Roar,
  Sing,
  Supersonic,
  SonicBoom,
  Disable,
  Acid,
  Ember,
  Flamethrower,
  Mist,
  WaterGun,
  HydroPump,
  Surf,
  IceBeam,
  Blizzard,
  Psybeam,
  BubbleBeam,
  AuroraBeam,
  HyperBeam,
  Peck,
  DrillPeck,
  Submission,
  LowKick,
  Counter,
  SeismicToss,
  Strength,
  Absorb,
  MegaDrain,
  LeechSeed,
  Growth,
  RazorLeaf,
  SolarBeam,
  PoisonPowder,
  StunSpore,
  SleepPowder,
  PetalDance,
  StringShot,
  DragonRage,
  FireSpin,
  ThunderShock,
  Thunderbolt,
  ThunderWave,
  Thunder,
  RockThrow,
  Earthquake,
  Fissure,
  Dig,
  Toxic,
  Confusion,
  Psychic,
  Hypnosis,
  Meditate,
  Agility,
  QuickAttack,
  Rage,
  Teleport,
  NightShade,
  Mimic,
  Screech,
  DoubleTeam,
  Recover,
  Harden,
  Minimize,
  Smokescreen,
  ConfuseRay,
  Withdraw,
  DefenseCurl,
  Barrier,
  LightScreen,
  Haze,
  Reflect,
  FocusEnergy,
  Bide,
  Metronome,
  MirrorMove,
  SelfDestruct,
  EggBomb,
  Lick,
  Smog,
  Sludge,
  BoneClub,
  FireBlast,
  Waterfall,
  Clamp,
  Swift,
  SkullBash,
  SpikeCannon,
  Constrict,
  Amnesia,
  Kinesis,
  SoftBoiled,
  HighJumpKick,
  Glare,
  DreamEater,
  PoisonGas,
  Barrage,
  LeechLife,
  LovelyKiss,
  SkyAttack,
  Transform,
  Bubble,
  DizzyPunch,
  Spore,
  Flash,
  Psywave,
  Splash,
  AcidArmor,
  Crabhammer,
  Explosion,
  FurySwipes,
  Bonemerang,
  Rest,
  RockSlide,
  HyperFang,
  Sharpen,
  Conversion,
  TriAttack,
  SuperFang,
  Slash,
  Substitute,
  Struggle
};

enum class Target : std::underlying_type_t<std::byte> {
  // none
  All,
  AllySide,
  Field,
  Self,
  // resolve
  AllOthers,
  Depends,
  Other,
  Any,
  Allies,
  Ally,
  AllyOrSelf,
  Foe,
  // resolve + run
  Foes,
  FoeSide,
  RandomFoe,
};

enum class Effect : std::underlying_type_t<std::byte> {
  None,
  // onBegin
  Confusion,
  Conversion,
  FocusEnergy,
  Haze,
  Heal,
  LeechSeed,
  LightScreen,
  Mimic,
  Mist,
  Paralyze,
  Poison,
  Reflect,
  Splash,
  Substitute,
  SwitchAndTeleport,
  Transform,
  // onEnd
  AccuracyDown1,
  AttackDown1,
  DefenseDown1,
  DefenseDown2,
  SpeedDown1,
  AttackUp1,
  AttackUp2,
  Bide,
  DefenseUp1,
  DefenseUp2,
  EvasionUp1,
  Sleep,
  SpecialUp1,
  SpecialUp2,
  SpeedUp2,
  // isSpecial
  DrainHP,
  DreamEater,
  Explode,
  JumpKick,
  PayDay,
  Rage,
  Recoil,
  Binding,
  Charge,
  SpecialDamage,
  SuperFang,
  Swift,
  Thrashing,
  // isMulti
  DoubleHit,
  MultiHit,
  Twineedle,
  // other
  AttackDownChance,
  DefenseDownChance,
  SpeedDownChance,
  SpecialDownChance,
  BurnChance1,
  BurnChance2,
  ConfusionChance,
  FlinchChance1,
  FlinchChance2,
  FreezeChance,
  ParalyzeChance1,
  ParalyzeChance2,
  PoisonChance1,
  PoisonChance2,
  Disable,
  HighCritical,
  HyperBeam,
  Metronome,
  MirrorMove,
  OHKO,
};

struct MoveData {
  Effect effect;
  uint8_t bp;
  Types type;
  uint8_t accuracy;
  Target target;
};

constexpr uint8_t percent(uint16_t p) { return (p * 0xFF) / 100; }

static constexpr std::array<MoveData, 165> MOVE_DATA{
    // Pound
    MoveData{
        Effect::None,
        40,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // KarateChop
    MoveData{
        Effect::HighCritical,
        50,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // DoubleSlap
    MoveData{
        Effect::MultiHit,
        15,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // CometPunch
    MoveData{
        Effect::MultiHit,
        18,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // MegaPunch
    MoveData{
        Effect::None,
        80,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // PayDay
    MoveData{
        Effect::PayDay,
        40,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // FirePunch
    MoveData{
        Effect::BurnChance1,
        75,
        Types::Fire,
        percent(100),
        Target::Other,
    },
    // IcePunch
    MoveData{
        Effect::FreezeChance,
        75,
        Types::Ice,
        percent(100),
        Target::Other,
    },
    // ThunderPunch
    MoveData{
        Effect::ParalyzeChance1,
        75,
        Types::Electric,
        percent(100),
        Target::Other,
    },
    // Scratch
    MoveData{
        Effect::None,
        40,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // ViseGrip
    MoveData{
        Effect::None,
        55,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Guillotine
    MoveData{
        Effect::OHKO,
        0,
        Types::Normal,
        percent(30),
        Target::Other,
    },
    // RazorWind
    MoveData{
        Effect::Charge,
        80,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // SwordsDance
    MoveData{
        Effect::AttackUp2,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Cut
    MoveData{
        Effect::None,
        50,
        Types::Normal,
        percent(95),
        Target::Other,
    },
    // Gust
    MoveData{
        Effect::None,
        40,
        Types::Normal,
        percent(100),
        Target::Any,
    },
    // WingAttack
    MoveData{
        Effect::None,
        35,
        Types::Flying,
        percent(100),
        Target::Any,
    },
    // Whirlwind
    MoveData{
        Effect::SwitchAndTeleport,
        0,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // Fly
    MoveData{
        Effect::Charge,
        70,
        Types::Flying,
        percent(95),
        Target::Any,
    },
    // Bind
    MoveData{
        Effect::Binding,
        15,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // Slam
    MoveData{
        Effect::None,
        80,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // VineWhip
    MoveData{
        Effect::None,
        35,
        Types::Grass,
        percent(100),
        Target::Other,
    },
    // Stomp
    MoveData{
        Effect::FlinchChance2,
        65,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // DoubleKick
    MoveData{
        Effect::DoubleHit,
        30,
        Types::Fighting,
        percent(100),
        Target::Other,
    },
    // MegaKick
    MoveData{
        Effect::None,
        120,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // JumpKick
    MoveData{
        Effect::JumpKick,
        70,
        Types::Fighting,
        percent(95),
        Target::Other,
    },
    // RollingKick
    MoveData{
        Effect::FlinchChance2,
        60,
        Types::Fighting,
        percent(85),
        Target::Other,
    },
    // SandAttack
    MoveData{
        Effect::AccuracyDown1,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Headbutt
    MoveData{
        Effect::FlinchChance2,
        70,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // HornAttack
    MoveData{
        Effect::None,
        65,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // FuryAttack
    MoveData{
        Effect::MultiHit,
        15,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // HornDrill
    MoveData{
        Effect::OHKO,
        0,
        Types::Normal,
        percent(30),
        Target::Other,
    },
    // Tackle
    MoveData{
        Effect::None,
        35,
        Types::Normal,
        percent(95),
        Target::Other,
    },
    // BodySlam
    MoveData{
        Effect::ParalyzeChance2,
        85,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Wrap
    MoveData{
        Effect::Binding,
        15,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // TakeDown
    MoveData{
        Effect::Recoil,
        90,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // Thrash
    MoveData{
        Effect::Thrashing,
        90,
        Types::Normal,
        percent(100),
        Target::RandomFoe,
    },
    // DoubleEdge
    MoveData{
        Effect::Recoil,
        100,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // TailWhip
    MoveData{
        Effect::DefenseDown1,
        0,
        Types::Normal,
        percent(100),
        Target::Foes,
    },
    // PoisonSting
    MoveData{
        Effect::PoisonChance1,
        15,
        Types::Poison,
        percent(100),
        Target::Other,
    },
    // Twineedle
    MoveData{
        Effect::Twineedle,
        25,
        Types::Bug,
        percent(100),
        Target::Other,
    },
    // PinMissile
    MoveData{
        Effect::MultiHit,
        14,
        Types::Bug,
        percent(85),
        Target::Other,
    },
    // Leer
    MoveData{
        Effect::DefenseDown1,
        0,
        Types::Normal,
        percent(100),
        Target::Foes,
    },
    // Bite
    MoveData{
        Effect::FlinchChance1,
        60,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Growl
    MoveData{
        Effect::AttackDown1,
        0,
        Types::Normal,
        percent(100),
        Target::Foes,
    },
    // Roar
    MoveData{
        Effect::SwitchAndTeleport,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Sing
    MoveData{
        Effect::Sleep,
        0,
        Types::Normal,
        percent(55),
        Target::Other,
    },
    // Supersonic
    MoveData{
        Effect::Confusion,
        0,
        Types::Normal,
        percent(55),
        Target::Other,
    },
    // SonicBoom
    MoveData{
        Effect::SpecialDamage,
        1,
        Types::Normal,
        percent(90),
        Target::Other,
    },
    // Disable
    MoveData{
        Effect::Disable,
        0,
        Types::Normal,
        percent(55),
        Target::Other,
    },
    // Acid
    MoveData{
        Effect::DefenseDownChance,
        40,
        Types::Poison,
        percent(100),
        Target::Other,
    },
    // Ember
    MoveData{
        Effect::BurnChance1,
        40,
        Types::Fire,
        percent(100),
        Target::Other,
    },
    // Flamethrower
    MoveData{
        Effect::BurnChance1,
        95,
        Types::Fire,
        percent(100),
        Target::Other,
    },
    // Mist
    MoveData{
        Effect::Mist,
        0,
        Types::Ice,
        percent(100),
        Target::Self,
    },
    // WaterGun
    MoveData{
        Effect::None,
        40,
        Types::Water,
        percent(100),
        Target::Other,
    },
    // HydroPump
    MoveData{
        Effect::None,
        120,
        Types::Water,
        percent(80),
        Target::Other,
    },
    // Surf
    MoveData{
        Effect::None,
        95,
        Types::Water,
        percent(100),
        Target::Foes,
    },
    // IceBeam
    MoveData{
        Effect::FreezeChance,
        95,
        Types::Ice,
        percent(100),
        Target::Other,
    },
    // Blizzard
    MoveData{
        Effect::FreezeChance,
        120,
        Types::Ice,
        percent(90),
        Target::Other,
    },
    // Psybeam
    MoveData{
        Effect::ConfusionChance,
        65,
        Types::Psychic,
        percent(100),
        Target::Other,
    },
    // BubbleBeam
    MoveData{
        Effect::SpeedDownChance,
        65,
        Types::Water,
        percent(100),
        Target::Other,
    },
    // AuroraBeam
    MoveData{
        Effect::AttackDownChance,
        65,
        Types::Ice,
        percent(100),
        Target::Other,
    },
    // HyperBeam
    MoveData{
        Effect::HyperBeam,
        150,
        Types::Normal,
        percent(90),
        Target::Other,
    },
    // Peck
    MoveData{
        Effect::None,
        35,
        Types::Flying,
        percent(100),
        Target::Any,
    },
    // DrillPeck
    MoveData{
        Effect::None,
        80,
        Types::Flying,
        percent(100),
        Target::Any,
    },
    // Submission
    MoveData{
        Effect::Recoil,
        80,
        Types::Fighting,
        percent(80),
        Target::Other,
    },
    // LowKick
    MoveData{
        Effect::FlinchChance2,
        50,
        Types::Fighting,
        percent(90),
        Target::Other,
    },
    // Counter
    MoveData{
        Effect::None,
        1,
        Types::Fighting,
        percent(100),
        Target::Depends,
    },
    // SeismicToss
    MoveData{
        Effect::SpecialDamage,
        1,
        Types::Fighting,
        percent(100),
        Target::Other,
    },
    // Strength
    MoveData{
        Effect::None,
        80,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Absorb
    MoveData{
        Effect::DrainHP,
        20,
        Types::Grass,
        percent(100),
        Target::Other,
    },
    // MegaDrain
    MoveData{
        Effect::DrainHP,
        40,
        Types::Grass,
        percent(100),
        Target::Other,
    },
    // LeechSeed
    MoveData{
        Effect::LeechSeed,
        0,
        Types::Grass,
        percent(90),
        Target::Other,
    },
    // Growth
    MoveData{
        Effect::SpecialUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // RazorLeaf
    MoveData{
        Effect::HighCritical,
        55,
        Types::Grass,
        percent(95),
        Target::Other,
    },
    // SolarBeam
    MoveData{
        Effect::Charge,
        120,
        Types::Grass,
        percent(100),
        Target::Other,
    },
    // PoisonPowder
    MoveData{
        Effect::Poison,
        0,
        Types::Poison,
        percent(75),
        Target::Other,
    },
    // StunSpore
    MoveData{
        Effect::Paralyze,
        0,
        Types::Grass,
        percent(75),
        Target::Other,
    },
    // SleepPowder
    MoveData{
        Effect::Sleep,
        0,
        Types::Grass,
        percent(75),
        Target::Other,
    },
    // PetalDance
    MoveData{
        Effect::Thrashing,
        70,
        Types::Grass,
        percent(100),
        Target::RandomFoe,
    },
    // StringShot
    MoveData{
        Effect::SpeedDown1,
        0,
        Types::Bug,
        percent(95),
        Target::Foes,
    },
    // DragonRage
    MoveData{
        Effect::SpecialDamage,
        1,
        Types::Dragon,
        percent(100),
        Target::Other,
    },
    // FireSpin
    MoveData{
        Effect::Binding,
        15,
        Types::Fire,
        percent(70),
        Target::Other,
    },
    // ThunderShock
    MoveData{
        Effect::ParalyzeChance1,
        40,
        Types::Electric,
        percent(100),
        Target::Other,
    },
    // Thunderbolt
    MoveData{
        Effect::ParalyzeChance1,
        95,
        Types::Electric,
        percent(100),
        Target::Other,
    },
    // ThunderWave
    MoveData{
        Effect::Paralyze,
        0,
        Types::Electric,
        percent(100),
        Target::Other,
    },
    // Thunder
    MoveData{
        Effect::ParalyzeChance1,
        120,
        Types::Electric,
        percent(70),
        Target::Other,
    },
    // RockThrow
    MoveData{
        Effect::None,
        50,
        Types::Rock,
        percent(65),
        Target::Other,
    },
    // Earthquake
    MoveData{
        Effect::None,
        100,
        Types::Ground,
        percent(100),
        Target::AllOthers,
    },
    // Fissure
    MoveData{
        Effect::OHKO,
        0,
        Types::Ground,
        percent(30),
        Target::Other,
    },
    // Dig
    MoveData{
        Effect::Charge,
        100,
        Types::Ground,
        percent(100),
        Target::Other,
    },
    // Toxic
    MoveData{
        Effect::Poison,
        0,
        Types::Poison,
        percent(85),
        Target::Other,
    },
    // Confusion
    MoveData{
        Effect::ConfusionChance,
        50,
        Types::Psychic,
        percent(100),
        Target::Other,
    },
    // Psychic
    MoveData{
        Effect::SpecialDownChance,
        90,
        Types::Psychic,
        percent(100),
        Target::Other,
    },
    // Hypnosis
    MoveData{
        Effect::Sleep,
        0,
        Types::Psychic,
        percent(60),
        Target::Other,
    },
    // Meditate
    MoveData{
        Effect::AttackUp1,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // Agility
    MoveData{
        Effect::SpeedUp2,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // QuickAttack
    MoveData{
        Effect::None,
        40,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Rage
    MoveData{
        Effect::Rage,
        20,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Teleport
    MoveData{
        Effect::SwitchAndTeleport,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // NightShade
    MoveData{
        Effect::SpecialDamage,
        1,
        Types::Ghost,
        percent(100),
        Target::Other,
    },
    // Mimic
    MoveData{
        Effect::Mimic,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Screech
    MoveData{
        Effect::DefenseDown2,
        0,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // DoubleTeam
    MoveData{
        Effect::EvasionUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Recover
    MoveData{
        Effect::Heal,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Harden
    MoveData{
        Effect::DefenseUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Minimize
    MoveData{
        Effect::EvasionUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Smokescreen
    MoveData{
        Effect::AccuracyDown1,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // ConfuseRay
    MoveData{
        Effect::Confusion,
        0,
        Types::Ghost,
        percent(100),
        Target::Other,
    },
    // Withdraw
    MoveData{
        Effect::DefenseUp1,
        0,
        Types::Water,
        percent(100),
        Target::Self,
    },
    // DefenseCurl
    MoveData{
        Effect::DefenseUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Barrier
    MoveData{
        Effect::DefenseUp2,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // LightScreen
    MoveData{
        Effect::LightScreen,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // Haze
    MoveData{
        Effect::Haze,
        0,
        Types::Ice,
        percent(100),
        Target::Self,
    },
    // Reflect
    MoveData{
        Effect::Reflect,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // FocusEnergy
    MoveData{
        Effect::FocusEnergy,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Bide
    MoveData{
        Effect::Bide,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Metronome
    MoveData{
        Effect::Metronome,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // MirrorMove
    MoveData{
        Effect::MirrorMove,
        0,
        Types::Flying,
        percent(100),
        Target::Self,
    },
    // SelfDestruct
    MoveData{
        Effect::Explode,
        130,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // EggBomb
    MoveData{
        Effect::None,
        100,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // Lick
    MoveData{
        Effect::ParalyzeChance2,
        20,
        Types::Ghost,
        percent(100),
        Target::Other,
    },
    // Smog
    MoveData{
        Effect::PoisonChance2,
        20,
        Types::Poison,
        percent(70),
        Target::Other,
    },
    // Sludge
    MoveData{
        Effect::PoisonChance2,
        65,
        Types::Poison,
        percent(100),
        Target::Other,
    },
    // BoneClub
    MoveData{
        Effect::FlinchChance1,
        65,
        Types::Ground,
        percent(85),
        Target::Other,
    },
    // FireBlast
    MoveData{
        Effect::BurnChance2,
        120,
        Types::Fire,
        percent(85),
        Target::Other,
    },
    // Waterfall
    MoveData{
        Effect::None,
        80,
        Types::Water,
        percent(100),
        Target::Other,
    },
    // Clamp
    MoveData{
        Effect::Binding,
        35,
        Types::Water,
        percent(75),
        Target::Other,
    },
    // Swift
    MoveData{
        Effect::Swift,
        60,
        Types::Normal,
        percent(100),
        Target::Foes,
    },
    // SkullBash
    MoveData{
        Effect::Charge,
        100,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // SpikeCannon
    MoveData{
        Effect::MultiHit,
        20,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Constrict
    MoveData{
        Effect::SpeedDownChance,
        10,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Amnesia
    MoveData{
        Effect::SpecialUp2,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // Kinesis
    MoveData{
        Effect::AccuracyDown1,
        0,
        Types::Psychic,
        percent(80),
        Target::Other,
    },
    // SoftBoiled
    MoveData{
        Effect::Heal,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // HighJumpKick
    MoveData{
        Effect::JumpKick,
        85,
        Types::Fighting,
        percent(90),
        Target::Other,
    },
    // Glare
    MoveData{
        Effect::Paralyze,
        0,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // DreamEater
    MoveData{
        Effect::DreamEater,
        100,
        Types::Psychic,
        percent(100),
        Target::Other,
    },
    // PoisonGas
    MoveData{
        Effect::Poison,
        0,
        Types::Poison,
        percent(55),
        Target::Other,
    },
    // Barrage
    MoveData{
        Effect::MultiHit,
        15,
        Types::Normal,
        percent(85),
        Target::Other,
    },
    // LeechLife
    MoveData{
        Effect::DrainHP,
        20,
        Types::Bug,
        percent(100),
        Target::Other,
    },
    // LovelyKiss
    MoveData{
        Effect::Sleep,
        0,
        Types::Normal,
        percent(75),
        Target::Other,
    },
    // SkyAttack
    MoveData{
        Effect::Charge,
        140,
        Types::Flying,
        percent(90),
        Target::Any,
    },
    // Transform
    MoveData{
        Effect::Transform,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Bubble
    MoveData{
        Effect::SpeedDownChance,
        20,
        Types::Water,
        percent(100),
        Target::Other,
    },
    // DizzyPunch
    MoveData{
        Effect::None,
        70,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Spore
    MoveData{
        Effect::Sleep,
        0,
        Types::Grass,
        percent(100),
        Target::Other,
    },
    // Flash
    MoveData{
        Effect::AccuracyDown1,
        0,
        Types::Normal,
        percent(70),
        Target::Other,
    },
    // Psywave
    MoveData{
        Effect::SpecialDamage,
        1,
        Types::Psychic,
        percent(80),
        Target::Other,
    },
    // Splash
    MoveData{
        Effect::Splash,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // AcidArmor
    MoveData{
        Effect::DefenseUp2,
        0,
        Types::Poison,
        percent(100),
        Target::Self,
    },
    // Crabhammer
    MoveData{
        Effect::HighCritical,
        90,
        Types::Water,
        percent(85),
        Target::Other,
    },
    // Explosion
    MoveData{
        Effect::Explode,
        170,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // FurySwipes
    MoveData{
        Effect::MultiHit,
        18,
        Types::Normal,
        percent(80),
        Target::Other,
    },
    // Bonemerang
    MoveData{
        Effect::DoubleHit,
        50,
        Types::Ground,
        percent(90),
        Target::Other,
    },
    // Rest
    MoveData{
        Effect::Heal,
        0,
        Types::Psychic,
        percent(100),
        Target::Self,
    },
    // RockSlide
    MoveData{
        Effect::None,
        75,
        Types::Rock,
        percent(90),
        Target::Other,
    },
    // HyperFang
    MoveData{
        Effect::FlinchChance1,
        80,
        Types::Normal,
        percent(90),
        Target::Other,
    },
    // Sharpen
    MoveData{
        Effect::AttackUp1,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Conversion
    MoveData{
        Effect::Conversion,
        0,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // TriAttack
    MoveData{
        Effect::None,
        80,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // SuperFang
    MoveData{
        Effect::SuperFang,
        1,
        Types::Normal,
        percent(90),
        Target::Other,
    },
    // Slash
    MoveData{
        Effect::HighCritical,
        70,
        Types::Normal,
        percent(100),
        Target::Other,
    },
    // Substitute
    MoveData{
        Effect::Substitute,
        0,
        Types::Normal,
        percent(100),
        Target::Self,
    },
    // Struggle
    MoveData{
        Effect::Recoil,
        50,
        Types::Normal,
        percent(100),
        Target::RandomFoe,
    }};

static constexpr std::array<uint8_t, 165> PP{
    35, // Pound
    25, // KarateChop
    10, // DoubleSlap
    15, // CometPunch
    20, // MegaPunch
    20, // PayDay
    15, // FirePunch
    15, // IcePunch
    15, // ThunderPunch
    35, // Scratch
    30, // ViseGrip
    5,  // Guillotine
    10, // RazorWind
    30, // SwordsDance
    30, // Cut
    35, // Gust
    35, // WingAttack
    20, // Whirlwind
    15, // Fly
    20, // Bind
    20, // Slam
    10, // VineWhip
    20, // Stomp
    30, // DoubleKick
    5,  // MegaKick
    25, // JumpKick
    15, // RollingKick
    15, // SandAttack
    15, // Headbutt
    25, // HornAttack
    20, // FuryAttack
    5,  // HornDrill
    35, // Tackle
    15, // BodySlam
    20, // Wrap
    20, // TakeDown
    20, // Thrash
    15, // DoubleEdge
    30, // TailWhip
    35, // PoisonSting
    20, // Twineedle
    20, // PinMissile
    30, // Leer
    25, // Bite
    40, // Growl
    20, // Roar
    15, // Sing
    20, // Supersonic
    20, // SonicBoom
    20, // Disable
    30, // Acid
    25, // Ember
    15, // Flamethrower
    30, // Mist
    25, // WaterGun
    5,  // HydroPump
    15, // Surf
    10, // IceBeam
    5,  // Blizzard
    20, // Psybeam
    20, // BubbleBeam
    20, // AuroraBeam
    5,  // HyperBeam
    35, // Peck
    20, // DrillPeck
    25, // Submission
    20, // LowKick
    20, // Counter
    20, // SeismicToss
    15, // Strength
    20, // Absorb
    10, // MegaDrain
    10, // LeechSeed
    40, // Growth
    25, // RazorLeaf
    10, // SolarBeam
    35, // PoisonPowder
    30, // StunSpore
    15, // SleepPowder
    20, // PetalDance
    40, // StringShot
    10, // DragonRage
    15, // FireSpin
    30, // ThunderShock
    15, // Thunderbolt
    20, // ThunderWave
    10, // Thunder
    15, // RockThrow
    10, // Earthquake
    5,  // Fissure
    10, // Dig
    10, // Toxic
    25, // Confusion
    10, // Psychic
    20, // Hypnosis
    40, // Meditate
    30, // Agility
    30, // QuickAttack
    20, // Rage
    20, // Teleport
    15, // NightShade
    10, // Mimic
    40, // Screech
    15, // DoubleTeam
    20, // Recover
    30, // Harden
    20, // Minimize
    20, // Smokescreen
    10, // ConfuseRay
    40, // Withdraw
    40, // DefenseCurl
    30, // Barrier
    30, // LightScreen
    30, // Haze
    20, // Reflect
    30, // FocusEnergy
    10, // Bide
    10, // Metronome
    20, // MirrorMove
    5,  // SelfDestruct
    10, // EggBomb
    30, // Lick
    20, // Smog
    20, // Sludge
    20, // BoneClub
    5,  // FireBlast
    15, // Waterfall
    10, // Clamp
    20, // Swift
    15, // SkullBash
    15, // SpikeCannon
    35, // Constrict
    20, // Amnesia
    15, // Kinesis
    10, // SoftBoiled
    20, // HighJumpKick
    30, // Glare
    15, // DreamEater
    40, // PoisonGas
    20, // Barrage
    15, // LeechLife
    10, // LovelyKiss
    5,  // SkyAttack
    10, // Transform
    30, // Bubble
    10, // DizzyPunch
    15, // Spore
    20, // Flash
    15, // Psywave
    40, // Splash
    40, // AcidArmor
    10, // Crabhammer
    5,  // Explosion
    15, // FurySwipes
    10, // Bonemerang
    10, // Rest
    10, // RockSlide
    15, // HyperFang
    30, // Sharpen
    30, // Conversion
    10, // TriAttack
    10, // SuperFang
    20, // Slash
    10, // Substitute
    10, // Struggle,
};

static_assert(sizeof(Move) == 1);

constexpr auto get_move_data(const auto move) noexcept {
  return MOVE_DATA[static_cast<uint8_t>(move) - 1];
}

constexpr uint8_t get_max_pp(const auto move) noexcept {
  return std::min(PP[static_cast<uint8_t>(move) - 1] / 5 * 8, 61);
}

} // namespace Data

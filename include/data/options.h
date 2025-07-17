#pragma once

namespace LibpkmnOptions {

#ifdef LOG
constexpr bool log = true;
#else
constexpr bool log = false;
#endif

#ifdef CHANCE
constexpr bool chance = true;
#else
constexpr bool chance = false;
#endif

#ifdef CALC
constexpr bool calc = true;
#else
constexpr bool calc = false;
#endif

} // namespace LibpkmnOptions

namespace Options {

// The would lower input dim for pokemon/active
namespace Moves {
constexpr bool allow_transform = true;
constexpr bool allow_metronome = true;
// duration stuff
constexpr bool allow_confusion = true;
constexpr bool allow_thrash = true;
constexpr bool allow_bide = true;
constexpr bool allow_disable = true;

} // namespace Moves

// smaller team nets?
namespace Pokemon {
constexpr bool allow_not_fully_evolved = true;
}

} // namespace Options
#pragma once

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
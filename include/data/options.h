#pragma once

namespace Options {

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

} // namespace Options

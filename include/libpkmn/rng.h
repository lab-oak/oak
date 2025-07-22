#pragma once

#include <cstddef>

namespace RNG {

inline void next(uint64_t &seed) noexcept {
  seed = 0x5D588B656C078965 * seed + 0x0000000000269EC3;
}

} // namespace RNG
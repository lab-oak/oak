#pragma once

#include <array>
#include <cstddef>

namespace Data {
constexpr std::array<std::array<uint8_t, 2>, 13> boosts{
    std::array<uint8_t, 2>{25, 100}, // -6
    {28, 100},                       // -5
    {33, 100},                       // -4
    {40, 100},                       // -3
    {50, 100},                       // -2
    {66, 100},                       // -1
    {1, 1},                          //  0
    {15, 10},                        // +1
    {2, 1},                          // +2
    {25, 10},                        // +3
    {3, 1},                          // +4
    {35, 10},                        // +5
    {4, 1}                           // +6
};
};
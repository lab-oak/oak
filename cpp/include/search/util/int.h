#pragma once

#pragma pack(push, 1)
struct uint24_t {
  std::array<uint8_t, 3> _data;

  constexpr uint24_t() = default;

  constexpr uint24_t(uint32_t value) noexcept {
    _data[0] = value & 0xFF;
    _data[1] = (value >> 8) & 0xFF;
    _data[2] = (value >> 16) & 0xFF;
  }

  constexpr operator uint32_t() const noexcept {
    return (static_cast<uint32_t>(_data[0]) |
            (static_cast<uint32_t>(_data[1]) << 8) |
            (static_cast<uint32_t>(_data[2]) << 16));
  }

  constexpr auto value() const noexcept { return static_cast<uint32_t>(*this); }

  constexpr uint24_t &operator++() noexcept {
    uint32_t value = static_cast<uint32_t>(*this) + 1;
    *this = uint24_t(value);
    return *this;
  }
};
#pragma pack(pop)

struct uint24_t_test {
  static_assert(sizeof(uint24_t) == 3);

  static consteval uint24_t overflow() {
    uint24_t x{};
    for (size_t i = 0; i < (1 << 24); ++i) {
      ++x;
    }
    return x;
  }
};
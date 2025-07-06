#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cstddef>

// Provides minimal std::vector interface around a std::array
// May offer better performance in some cases
template <std::size_t max_size> struct ArrayBasedVector {
  template <typename T, typename CapacityT = std::size_t> class Vector {
  protected:
    std::array<T, max_size> _storage;
    CapacityT _size;

  public:
    constexpr Vector() : _size{} {}

    template <typename InT>
      requires(std::is_integral_v<InT>)
    constexpr Vector(const InT n) {
      assert(0 < n && n <= max_size);
      _size = n;
      std::fill(this->begin(), this->end(), T{});
    }

    template <typename Vec> constexpr Vector(const Vec &other) noexcept {
      assert(other.size() <= max_size);
      _size = other.size();
      std::copy(other.begin(), other.end(), _storage.begin());
    }

    template <typename Vec>
    constexpr Vector &operator=(const Vec &other) noexcept {
      assert(other.size() <= max_size);
      _size = other.size();
      std::copy(other.begin(), other.end(), _storage.begin());
      return *this;
    }

    template <typename Vec> bool operator==(const Vec &other) const noexcept {
      for (CapacityT i = 0; i < _size; ++i) {
        if ((*this)[i] != other[i]) {
          return false;
        }
      }
      return _size == other.size();
    }

    template <typename size_type>
    constexpr void resize(size_type n, T val = T{}) {
      assert(n <= max_size);
      if (_size < n) {
        std::fill(_storage.begin() + _size, _storage.begin() + n, val);
      }
      _size = n;
    }

    template <typename size_type> void reserve(size_type n) noexcept {
      assert(n <= max_size);
      _size = n;
    }

    constexpr void push_back(const T &val = T{}) {
      assert(_size < max_size);
      _storage[_size++] = val;
    }

    constexpr void push_back(T &&val = T{}) {
      assert(_size < max_size);
      _storage[_size++] = val;
    }

    constexpr T &operator[](auto n) { return _storage[n]; }

    constexpr const T &operator[](auto n) const { return _storage[n]; }

    CapacityT size() const noexcept { return _size; }

    constexpr void clear() noexcept { _size = 0; }

    constexpr auto begin() noexcept { return _storage.begin(); }

    constexpr const auto begin() const noexcept { return _storage.begin(); }

    constexpr auto end() noexcept { return _storage.begin() + _size; }

    const auto end() const noexcept { return _storage.begin() + _size; }
  };
};

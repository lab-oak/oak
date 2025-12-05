/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Input features and network structure used in NNUE evaluation function

#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <memory>

#include <nn/affine.h>
#include <nn/battle/stockfish/affine.h>
#include <nn/battle/stockfish/clipped_relu.h>
#include <nn/battle/stockfish/common.h>

namespace NN::Battle::Stockfish {

struct Network {
  virtual float
  propagate(const TransformedFeatureType *transformedFeatures) const = 0;
  virtual void copy_parameters(const Affine<> &, const Affine<> &,
                               const Affine<false> &) = 0;
};

template <int In, int Out1, int Out2> struct NetworkArchitecture : Network {
  static constexpr IndexType ConcatenatedSidesDims = In;
  static constexpr int FC_0_OUTPUTS = Out1;
  static constexpr int FC_1_OUTPUTS = Out2;

  AffineTransform<ConcatenatedSidesDims, FC_0_OUTPUTS> fc_0;
  ClippedReLU<FC_0_OUTPUTS> ac_0;
  AffineTransform<FC_0_OUTPUTS, FC_1_OUTPUTS> fc_1;
  ClippedReLU<FC_1_OUTPUTS> ac_1;
  AffineTransform<FC_1_OUTPUTS, 1> fc_2;

  void copy_parameters(const Affine<> &affine0, const Affine<> &affine1,
                       const Affine<false> &affine2) {
    fc_0.copy_parameters(affine0);
    fc_1.copy_parameters(affine1);
    fc_2.copy_parameters(affine2);
  }

  float
  propagate(const TransformedFeatureType *transformedFeatures) const override {
    struct alignas(CacheLineSize) Buffer {
      alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
      alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
      alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

      Buffer() { std::memset(this, 0, sizeof(*this)); }
    };

#if defined(__clang__) && (__APPLE__)
    // workaround for a bug reported with xcode 12
    static thread_local auto tlsBuffer = std::make_shared<Buffer>();
    // Access TLS only once, cache result.
    Buffer &buffer = *tlsBuffer;
#else
    alignas(CacheLineSize) static thread_local Buffer buffer;
#endif

    fc_0.propagate(transformedFeatures, buffer.fc_0_out);
    ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
    fc_1.propagate(buffer.ac_0_out, buffer.fc_1_out);
    ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
    fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

    // buffer.fc_0_out[FC_0_OUTPUTS] is such that 1.0 is equal to
    // 127*(1<<WeightScaleBits) in quantized form, but we want 1.0 to be equal
    // to 600*OutputScale
    auto outputValue = buffer.fc_2_out[0] / float(127 * (1 << 6));

    return outputValue;
  }
};

namespace Impl {
std::shared_ptr<Network> invalid(const std::string &msg) {
  throw std::runtime_error{"Invalid layer size for quantized net " + msg +
                           " (check code for valid sizes)."};
  return std::shared_ptr<Network>{nullptr};
}
template <int In, int H1> std::shared_ptr<Network> make_network_2(int h2) {
  if (H1 < h2) {
    throw std::runtime_error{"Quantized net must have decreasing layer size."};
    return std::shared_ptr<Network>{nullptr};
  }

  switch (h2) {
  case 32:
    return std::make_shared<NetworkArchitecture<In, H1, 32>>();
  case 64:
    return std::make_shared<NetworkArchitecture<In, H1, 64>>();
  default:
    return Impl::invalid("H2: " + std::to_string(h2));
  }
}
template <int In> std::shared_ptr<Network> make_network_1(int h1, int h2) {
  switch (h1) {
  case 32:
    return make_network_2<In, 32>(h2);
  case 64:
    return make_network_2<In, 64>(h2);
  case 128:
    return make_network_2<In, 128>(h2);
  default:
    return Impl::invalid("H1: " + std::to_string(h1));
  }
}
} // namespace Impl

std::shared_ptr<Network> make_network(int in, int h1, int h2) {
  switch (in) {
  case 512:
    return Impl::make_network_1<512>(h1, h2);
  case 768:
    return Impl::make_network_1<768>(h1, h2);
  case 1024:
  default:
    return Impl::invalid("Side dim: " + std::to_string(in));
  }
}

} // namespace NN::Battle::Stockfish
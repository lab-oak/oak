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

// Definition of layer AffineTransform of NNUE evaluation function

#pragma once

#include <cstdint>
#include <iostream>

#include "common.h"
#include "simd.h"

/*
  This file contains the definition for a fully connected layer (aka affine
  transform).

    - expected use-case is for when PaddedInputDimensions == 32 and
  InputDimensions <= 32.
      - that's why AVX512 is hard to implement
    - expected use-case is small layers
    - inputs are processed in chunks of 4, weights are respectively transposed
    - accumulation happens directly to int32s
*/

namespace NN::Battle::Stockfish {

template <IndexType InDims, IndexType OutDims> class AffineTransform {
public:
  // Input/output type
  using InputType = std::uint8_t;
  using OutputType = std::int32_t;

  // Number of input/output dimensions
  static constexpr IndexType InputDimensions = InDims;
  static constexpr IndexType OutputDimensions = OutDims;

  static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
  static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

  using OutputBuffer = OutputType[PaddedOutputDimensions];

  static constexpr IndexType get_weight_index_scrambled(IndexType i) {
    return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4 +
           i / PaddedInputDimensions * 4 + i % 4;
  }

  static constexpr IndexType get_weight_index(IndexType i) {
    return get_weight_index_scrambled(i);
  }

  void copy_parameters(const auto &affine) {
    assert(InputDimensions == affine.in_dim);
    assert(OutputDimensions == affine.out_dim);
    for (auto i = 0; i < OutputDimensions; ++i) {
      biases[i] = static_cast<int32_t>(affine.biases.data()[i] * 64 * 127);
    }
    for (auto i = 0; i < OutputDimensions * InputDimensions; ++i) {
      weights[get_weight_index(i)] = static_cast<int8_t>(affine.weights.data()[i] * 64);
      assert(affine.weights.data()[i] < 2);
      assert(affine.weights.data()[i] > -2);
    }
    // assert(std::none_of());
  }

  // Forward propagation
  void propagate(const InputType *input, OutputType *output) const {

    if constexpr (OutputDimensions > 1) {

      using vec_t = __m256i;
#define vec_set_32 _mm256_set1_epi32
#define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32

      static constexpr IndexType OutputSimdWidth =
          sizeof(vec_t) / sizeof(OutputType);

      static_assert(OutputDimensions % OutputSimdWidth == 0);

      constexpr IndexType NumChunks =
          ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
      constexpr IndexType NumRegs = OutputDimensions / OutputSimdWidth;

      const auto input32 = reinterpret_cast<const std::int32_t *>(input);
      const vec_t *biasvec = reinterpret_cast<const vec_t *>(biases);
      vec_t acc[NumRegs];
      for (IndexType k = 0; k < NumRegs; ++k)
        acc[k] = biasvec[k];

      for (IndexType i = 0; i < NumChunks; ++i) {
        const vec_t in0 = vec_set_32(input32[i]);
        const auto col0 =
            reinterpret_cast<const vec_t *>(&weights[i * OutputDimensions * 4]);

        for (IndexType k = 0; k < NumRegs; ++k)
          vec_add_dpbusd_32(acc[k], in0, col0[k]);
      }

      vec_t *outptr = reinterpret_cast<vec_t *>(output);
      for (IndexType k = 0; k < NumRegs; ++k)
        outptr[k] = acc[k];

#undef vec_set_32
#undef vec_add_dpbusd_32
    } else if constexpr (OutputDimensions == 1) {
      // We cannot use AVX512 for the last layer because there are only 32
      // inputs and the buffer is not padded to 64 elements.
      using vec_t = __m256i;
#define vec_setzero() _mm256_setzero_si256()
#define vec_set_32 _mm256_set1_epi32
#define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
#define vec_hadd Simd::m256_hadd

      const auto inputVector = reinterpret_cast<const vec_t *>(input);

      static constexpr IndexType InputSimdWidth =
          sizeof(vec_t) / sizeof(InputType);

      static_assert(PaddedInputDimensions % InputSimdWidth == 0);

      constexpr IndexType NumChunks = PaddedInputDimensions / InputSimdWidth;
      vec_t sum0 = vec_setzero();
      const auto row0 = reinterpret_cast<const vec_t *>(&weights[0]);

      for (int j = 0; j < int(NumChunks); ++j) {
        const vec_t in = inputVector[j];
        vec_add_dpbusd_32(sum0, in, row0[j]);
      }
      output[0] = vec_hadd(sum0, biases[0]);

#undef vec_setzero
#undef vec_set_32
#undef vec_add_dpbusd_32
#undef vec_hadd
    }
  }

public:
  using BiasType = OutputType;
  using WeightType = std::int8_t;

  alignas(CacheLineSize) BiasType biases[OutputDimensions];
  alignas(CacheLineSize)
      WeightType weights[OutputDimensions * PaddedInputDimensions];
};

} // namespace NN::Battle::Stockfish
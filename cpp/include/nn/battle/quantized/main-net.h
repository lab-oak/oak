#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <memory>

#include <nn/affine.h>
#include <nn/battle/quantized/affine.h>
#include <nn/battle/quantized/clipped_relu.h>
#include <nn/battle/quantized/common.h>

namespace NN::Battle::Quantized {

template <int In, int Out1, int Out2> struct MainNet {

  using T = uint8_t;

  static constexpr int FC_0_OUTPUTS = Out1;
  static constexpr int FC_1_OUTPUTS = Out2;

  AffineTransform<In, FC_0_OUTPUTS> fc_0;
  ClippedReLU<FC_0_OUTPUTS> ac_0;
  AffineTransform<FC_0_OUTPUTS, FC_0_OUTPUTS> fc_1;
  ClippedReLU<FC_0_OUTPUTS> ac_1;
  AffineTransform<FC_0_OUTPUTS, FC_1_OUTPUTS> fc_2;
  ClippedReLU<FC_1_OUTPUTS> ac_2;
  AffineTransform<FC_1_OUTPUTS, 1> fc_3;

  float propagate(const uint8_t *battle_embedding) const override {
    struct alignas(CacheLineSize) Buffer {
      alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
      alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
      alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
      alignas(CacheLineSize) typename decltype(ac_2)::OutputBuffer ac_2_out;
      alignas(CacheLineSize) typename decltype(fc_3)::OutputBuffer fc_3_out;

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

    fc_0.propagate(battle_embedding, buffer.fc_0_out);
    ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
    fc_1.propagate(buffer.ac_0_out, buffer.fc_1_out);
    ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
    fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);
    ac_2.propagate(buffer.fc_2_out, buffer.ac_2_out);
    fc_3.propagate(buffer.ac_2_out, buffer.fc_3_out);

    // buffer.fc_0_out[FC_0_OUTPUTS] is such that 1.0 is equal to
    // 127*(1<<WeightScaleBits) in quantized form, but we want 1.0 to be equal
    // to 600*OutputScale
    auto outputValue = buffer.fc_3_out[0] / float(127 * (1 << 6));
    return outputValue;
  }
};

} // namespace NN::Battle::Quantized
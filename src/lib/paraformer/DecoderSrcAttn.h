

#ifndef PARAFORMER_DECODERSRCATTN_H
#define PARAFORMER_DECODERSRCATTN_H

#include <stdint.h>

#include "../Tensor.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paraformer;

namespace paraformer {

class DecoderSrcAttn {
  private:
    DecSelfAttnParams *params;

    void linear_forward(Tensor<float> *din, Tensor<float> *dout, float *weight,
                        float *bias);

  public:
    DecoderSrcAttn(DecSelfAttnParams *params);
    ~DecoderSrcAttn();
    void forward(Tensor<float> *&din, Tensor<float> *enc);
};

} // namespace paraformer

#endif

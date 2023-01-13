

#ifndef PARAFORMER_SUBENCODER_H
#define PARAFORMER_SUBENCODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paraformer;
namespace paraformer {
class SubEncoder {
  private:
    SubEncoderParams *params;
    LayerNorm *norm1;
    LayerNorm *norm2;
    EncSelfAttn *self_attn;
    FeedForward *feedforward;

  public:
    SubEncoder(SubEncoderParams *params, int size);
    ~SubEncoder();
    void reset();
    void forward(Tensor<float> *&din, int *conv_im2col);
};
} // namespace paraformer

#endif

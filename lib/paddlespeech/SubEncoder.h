

#ifndef PS_SUBENCODER_H
#define PS_SUBENCODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ConvModule.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paddlespeech;

namespace paddlespeech {

class SubEncoder {
  private:
    Tensor<float> *in_cache;
    SubEncoderParams *params;
    ConvModule *conv_module;
    EncSelfAttn *self_attn;
    FeedForward *feedforward;
    FeedForward *feedforward_macron;
    LayerNorm *norm_ff;
    LayerNorm *norm_mha;
    LayerNorm *norm_macaron;
    LayerNorm *norm_conv;
    LayerNorm *norm_final;

  public:
    SubEncoder(SubEncoderParams *params, int mode);
    ~SubEncoder();
    void reset();
    void forward(Tensor<float> *din, Tensor<float> *pe);
};

} // namespace paddlespeech

#endif

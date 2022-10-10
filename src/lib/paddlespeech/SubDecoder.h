
#ifndef PS_SUBDECODER_H
#define PS_SUBDECODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ConvModule.h"
#include "DecSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paddlespeech;

namespace paddlespeech {

class SubDecoder {
  private:
    SubDecoderParams *params;
    // ConvModule *conv_module;
    //
    DecSelfAttn *self_attn;
    DecSelfAttn *src_attn;
    FeedForward *feedforward;
    // FeedForward *feedforward_macron;
    LayerNorm *norm1;
    LayerNorm *norm2;
    LayerNorm *norm3;

  public:
    SubDecoder(SubDecoderParams *params);
    ~SubDecoder();
    void forward(Tensor<float> *din, Tensor<int> *din_mask,
                 Tensor<float> *encoder_out, Tensor<int> *encoder_mask);
};

} // namespace paddlespeech

#endif

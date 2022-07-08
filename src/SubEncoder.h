

#ifndef SUBENCODER_H
#define SUBENCODER_H

#include <stdint.h>

#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "WenetParams.h"
#include "ConvModule.h"

class SubEncoder {
  private:
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
    SubEncoder(SubEncoderParams *params);
    ~SubEncoder();
    void forward(Tensor<float> *din, Tensor<float> *pe);
};

#endif

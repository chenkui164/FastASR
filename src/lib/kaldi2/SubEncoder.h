

#ifndef K2_SUBENCODER_H
#define K2_SUBENCODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ConvModule.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "ModelParams.h"

using namespace kaldi2;
namespace kaldi2 {

class SubEncoder {
  private:
    Tensor<float> *in_cache;
    SubEncoderParams *params;
    ConvModule *conv_module;
    EncSelfAttn *self_attn;
    FeedForward *feedforward;
    FeedForward *feedforward_macron;

  public:
    SubEncoder(SubEncoderParams *params, int mode);
    ~SubEncoder();
    void reset();
    void forward(Tensor<float> *din, Tensor<float> *pe);
};

} // namespace kaldi2
#endif

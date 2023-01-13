
#ifndef PARAFORMER_SUBDECODER_H
#define PARAFORMER_SUBDECODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "DecoderSrcAttn.h"
#include "FeedForwardDecoder.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paraformer;

namespace paraformer {

class SubDecoder {
  private:
    SubDecoderParams *params;
    FeedForwardDecoder *feedforward;
    DecoderSrcAttn *src_attn;
    LayerNorm *norm1;
    LayerNorm *norm2;
    LayerNorm *norm3;
    void forward_fsmn(Tensor<float> *din, int *conv_im2col);

  public:
    SubDecoder(SubDecoderParams *params);
    ~SubDecoder();
    void forward(Tensor<float> *din, Tensor<float> *enc, int *conv_im2col);
};
} // namespace paraformer

#endif

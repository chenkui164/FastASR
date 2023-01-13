

#ifndef PARAFORMER_DECODER_H
#define PARAFORMER_DECODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "FeedForwardDecoder.h"
#include "LayerNorm.h"
#include "ModelParams.h"
#include "SubDecoder.h"

using namespace paraformer;

namespace paraformer {

class Decoder {
  private:
    DecoderParams *params;
    SubDecoder *sub_decoders[16];
    LayerNorm *decoder3_norm;
    LayerNorm *after_norm;
    FeedForwardDecoder *feedforward;

    int *conv_im2col;

    void get_conv_im2col(int mm);

  public:
    Decoder(DecoderParams *params);
    ~Decoder();
    void forward(Tensor<float> *&din, Tensor<float> *enc);
};
} // namespace paraformer

#endif

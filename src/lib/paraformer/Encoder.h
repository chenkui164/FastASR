
#ifndef PARAFORMER_ENCODER_H
#define PARAFORMER_ENCODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "EmbedLayer.h"
#include "ModelParams.h"
#include "SubEncoder.h"

using namespace paraformer;

namespace paraformer {
class Encoder {
  private:
    int *conv_im2col;
    EncoderParams *params;
    SubEncoder *encoders0;
    SubEncoder *encoders[49];
    LayerNorm *after_norm;
    void get_conv_im2col(int mm);

    void get_poscode(Tensor<float> *poscode);
    void disp_conv_im2col(int mm);

  public:
    Encoder(EncoderParams *params);
    ~Encoder();
    void reset();
    void forward(Tensor<float> *&din);
};
} // namespace paraformer

#endif

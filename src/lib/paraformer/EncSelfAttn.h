

#ifndef PARAFORMER_ENCSELFATTN_H
#define PARAFORMER_ENCSELFATTN_H

#include <stdint.h>

#include "../Tensor.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paraformer;

namespace paraformer {

class EncSelfAttn {
  private:
    EncSelfAttnParams *params;
    void forward_fsmn(Tensor<float> *din, int *conv_im2col);
    void linear_qkv_forward(Tensor<float> *din, Tensor<float> *dout,
                            float *weight, float *bias);

  public:
    EncSelfAttn(EncSelfAttnParams *params);
    ~EncSelfAttn();
    void forward(Tensor<float> *&din, Tensor<float> *&mem, int *conv_im2col);
};

} // namespace paraformer

#endif

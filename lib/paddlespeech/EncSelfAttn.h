

#ifndef PS_ENCSELFATTN_H
#define PS_ENCSELFATTN_H

#include <stdint.h>

#include "../Tensor.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paddlespeech;
class EncSelfAttn {
  private:
    EncSelfAttnParams *params;

  public:
    EncSelfAttn(EncSelfAttnParams *params);
    ~EncSelfAttn();
    void forward(Tensor<float> *query, Tensor<float> *key, Tensor<float> *value,
                 Tensor<float> *pe);
};

#endif

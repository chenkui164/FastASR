
#ifndef PS_DECSELFATTN_H
#define PS_DECSELFATTN_H

#include <stdint.h>

#include "FeedForward.h"
#include "LayerNorm.h"
#include "../Tensor.h"
#include "ModelParams.h"

using namespace paddlespeech;
class DecSelfAttn {
  private:
    SelfAttnParams *params;

  public:
    DecSelfAttn(SelfAttnParams *params);
    ~DecSelfAttn();
    void forward(Tensor<float> *&query, Tensor<float> *key,
                 Tensor<float> *value, Tensor<int> *mask);
};

#endif

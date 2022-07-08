
#ifndef DECSELFATTN_H
#define DECSELFATTN_H

#include <stdint.h>

#include "FeedForward.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "WenetParams.h"

class DecSelfAttn {
  private:
    SelfAttnParams *params;

  public:
    DecSelfAttn(SelfAttnParams *params);
    ~DecSelfAttn();
    void forward(Tensor<float> *query, Tensor<float> *key, Tensor<float> *value,
                 Tensor<int> *mask, Tensor<float> *dout);
};

#endif

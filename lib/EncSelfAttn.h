

#ifndef ENCSELFATTN_H
#define ENCSELFATTN_H

#include <stdint.h>

#include "FeedForward.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "WenetParams.h"

class EncSelfAttn {
  private:
    EncSelfAttnParams *params;

  public:
    EncSelfAttn(EncSelfAttnParams *params);
    ~EncSelfAttn();
    void forward(Tensor<float> *query, Tensor<float> *key, Tensor<float> *value, Tensor<float> *pe,
                 Tensor<float> *dout);
};

#endif

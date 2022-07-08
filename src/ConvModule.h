

#ifndef CONVMODULE_H
#define CONVMODULE_H

#include <stdint.h>

#include "FeedForward.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "WenetParams.h"

class ConvModule {
  private:
    LayerNorm *norm;
    EncConvParams *params;

  public:
    ConvModule(EncConvParams *params);
    ~ConvModule();
    void forward(Tensor<float> *din);
};

#endif

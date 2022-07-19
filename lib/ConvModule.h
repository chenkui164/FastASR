

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
    Tensor<float> *conv_cache;
    int mode;
    void forward_mode0(Tensor<float> *din);
    void forward_mode1(Tensor<float> *din);

  public:
    ConvModule(EncConvParams *params, int mode);
    ~ConvModule();
    void reset();
    void forward(Tensor<float> *din);
};

#endif



#ifndef K2_CONVMODULE_H
#define K2_CONVMODULE_H

#include <stdint.h>

#include "../Tensor.h"
#include "FeedForward.h"
#include "ModelParams.h"

using namespace kaldi2;

namespace kaldi2 {
class ConvModule {
  private:
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
} // namespace kaldi2

#endif

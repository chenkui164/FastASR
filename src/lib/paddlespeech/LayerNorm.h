

#ifndef PS_LAYERNORM_H
#define PS_LAYERNORM_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paddlespeech;
class LayerNorm {
  private:
    NormParams *params;
    float error;
    void mean_var(float *din, float &mean, float &var);
    void norm(float *din, float mean, float var);

  public:
    LayerNorm(NormParams *params, float error);
    ~LayerNorm();
    void forward(Tensor<float> *din);
};

#endif

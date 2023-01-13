

#ifndef PARAFORMER_LAYERNORM_H
#define PARAFORMER_LAYERNORM_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paraformer;

namespace paraformer {
class LayerNorm {
  private:
    NormParams *params;
    float error;
    int layer_size;
    void mean_var(float *din, float &mean, float &var);
    void norm(float *din, float mean, float var);

  public:
    LayerNorm(NormParams *params, float error, int layer_size);
    ~LayerNorm();
    void forward(Tensor<float> *din);
};
} // namespace paraformer

#endif

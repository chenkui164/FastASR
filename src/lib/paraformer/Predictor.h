
#ifndef PARAFORMER_PREDICTOR_H
#define PARAFORMER_PREDICTOR_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paraformer;

namespace paraformer {

class Predictor {
  private:
    PredictorParams *params;
    void cif_conv1d(Tensor<float> *&din);
    int *conv_im2col;

    void get_conv_im2col(int mm);
    void disp_conv_im2col(int mm);

  public:
    Predictor(PredictorParams *params);
    ~Predictor();
    void forward(Tensor<float> *&din);
};
} // namespace paraformer

#endif



#ifndef PS_DECEMBEDLAYER_H
#define PS_DECEMBEDLAYER_H

#include <stdint.h>

#include "../Tensor.h"
#include "PositionEncoding.h"
#include "ModelParams.h"

using namespace paddlespeech;
class DecEmbedLayer {
  private:
    float *params;

  public:
    DecEmbedLayer(float *params);
    ~DecEmbedLayer();
    void forward(Tensor<int> *din, Tensor<float> *&dout);
};

#endif

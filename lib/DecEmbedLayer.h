

#ifndef DECEMBEDLAYER_H
#define DECEMBEDLAYER_H

#include <stdint.h>

#include "PositionEncoding.h"
#include "Tensor.h"
#include "WenetParams.h"

class DecEmbedLayer {
  private:
    float *params;

  public:
    DecEmbedLayer(float *params);
    ~DecEmbedLayer();
    void forward(Tensor<int> *din, Tensor<float> *&dout);
};

#endif

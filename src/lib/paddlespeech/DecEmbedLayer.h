

#ifndef PS_DECEMBEDLAYER_H
#define PS_DECEMBEDLAYER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"
#include "PositionEncoding.h"

using namespace paddlespeech;

namespace paddlespeech {

class DecEmbedLayer {
  private:
    float *params;

  public:
    DecEmbedLayer(float *params);
    ~DecEmbedLayer();
    void forward(Tensor<int> *din, Tensor<float> *&dout);
};

} // namespace paddlespeech
#endif

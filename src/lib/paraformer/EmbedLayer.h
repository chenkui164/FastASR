
#ifndef PARAFORMER_EMBEDLAYER_H
#define PARAFORMER_EMBEDLAYER_H

#include <stdint.h>

#include "../Tensor.h"

namespace paraformer {

class EmbedLayer {
  private:
    void get_poscode(int nn);

  public:
    EmbedLayer();
    ~EmbedLayer();
    void forward(Tensor<float> *din);
};

} // namespace paraformer

#endif

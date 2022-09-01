
#ifndef PS_EMBEDLAYER_H
#define PS_EMBEDLAYER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paddlespeech;
class EmbedLayer {
  private:
    EncEmbedParams *params;
    void get_conv_ind(int trans, int in_row, int in_column, int kernel,
                      int stride, int &out_row, int &out_column,
                      int *&out_idxs);
    void conv0_forward(Tensor<float> *&din);
    void conv1_forward(Tensor<float> *&din);
    void linear_out_forward(Tensor<float> *&din);

  public:
    EmbedLayer(EncEmbedParams *params);
    ~EmbedLayer();
    void forward(Tensor<float> *&din);
};

#endif

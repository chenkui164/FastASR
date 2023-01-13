
#ifndef PARAFORMER_FEEDFORWARDDECODER_H
#define PARAFORMER_FEEDFORWARDDECODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "LayerNorm.h"
#include "ModelParams.h"

using namespace paraformer;
namespace paraformer {

class FeedForwardDecoder {
  private:
    DecoderFeedForwardParams *params;
    void (*activate)(Tensor<float> *din);
    LayerNorm *norm;

  public:
    FeedForwardDecoder(DecoderFeedForwardParams *params);
    ~FeedForwardDecoder();
    void forward(Tensor<float> *din);
};

} // namespace paraformer

#endif

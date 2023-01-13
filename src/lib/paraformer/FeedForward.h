

#ifndef PARAFORMER_FEEDFORWARD_H
#define PARAFORMER_FEEDFORWARD_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paraformer;
namespace paraformer {

class FeedForward {
  private:
    FeedForwardParams *params;
    void (*activate)(Tensor<float> *din);

  public:
    FeedForward(FeedForwardParams *params, int active_type);
    ~FeedForward();
    void forward(Tensor<float> *din);
};

} // namespace paraformer

#endif

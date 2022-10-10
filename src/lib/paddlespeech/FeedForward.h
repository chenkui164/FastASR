

#ifndef PS_FEEDFORWARD_H
#define PS_FEEDFORWARD_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace paddlespeech;
namespace paddlespeech {

class FeedForward {
  private:
    FeedForwardParams *params;
    void (*activate)(Tensor<float> *din);

  public:
    FeedForward(FeedForwardParams *params, int active_type);
    ~FeedForward();
    void forward(Tensor<float> *din);
};

} // namespace paddlespeech

#endif

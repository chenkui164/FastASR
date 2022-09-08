

#ifndef K2_FEEDFORWARD_H
#define K2_FEEDFORWARD_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace kaldi2;
namespace kaldi2 {

class FeedForward {
  private:
    FeedForwardParams *params;
    void (*activate)(Tensor<float> *din);

  public:
    FeedForward(FeedForwardParams *params);
    ~FeedForward();
    void forward(Tensor<float> *din);
};

} // namespace kaldi2

#endif

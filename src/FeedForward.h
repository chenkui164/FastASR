

#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <stdint.h>

#include "Tensor.h"
#include "WenetParams.h"

class FeedForward {
  private:
    FeedForwardParams *params;
    void (*activate)(Tensor<float> *din);

  public:
    FeedForward(FeedForwardParams *params, int active_type);
    ~FeedForward();
    void forward(Tensor<float> *din);
};

#endif

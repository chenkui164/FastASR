
#ifndef FEATUREQUEUE_H
#define FEATUREQUEUE_H

#include "Tensor.h"
#include <queue>
#include <stdint.h>
using namespace std;

enum SpeechFlag { S_BEGIN, S_MIDDLE, S_END, S_ALL, S_ERR };

class FeatureQueue {
  private:
    queue<Tensor<float> *> feature_queue;
    Tensor<float> *buff;
    int buff_idx;
    int window_size;

  public:
    FeatureQueue();
    ~FeatureQueue();
    void reinit(int size);
    void reset();
    void push(float *din, SpeechFlag flag);
    Tensor<float> *pop();
    int size();
};

#endif

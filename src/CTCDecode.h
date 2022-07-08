
#ifndef CTCDECODE_H
#define CTCDECODE_H

#include <deque>
#include <math.h>
#include <stdint.h>
#include <vector>

#include "Tensor.h"
#include "WenetParams.h"

using namespace std;

struct CharProb {
    int char_idx;
    float prob;
};

struct PathProb {
    vector<int> prefix;
    float pb = -INFINITY;
    float pnb = -INFINITY;
    float prob = -INFINITY;
};

class CTCdecode {
  private:
    float *ctc_weight;
    float *ctc_bias;

  public:
    CTCdecode(float *ctc_weight, float *ctc_bias);
    ~CTCdecode();
    void forward(Tensor<float> *din, deque<PathProb> &hyps);
    void show_hyps(deque<PathProb> hyps);
};

#endif

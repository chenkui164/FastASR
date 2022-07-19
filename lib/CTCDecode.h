
#ifndef CTCDECODE_H
#define CTCDECODE_H

#include <deque>
#include <math.h>
#include <set>
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
    float v_s = -INFINITY;
    float v_ns = -INFINITY;
    float cur_token_prob = -INFINITY;
    vector<int> times_s;
    vector<int> times_ns;
};

// auto path_cmp = [](PathProb a, PathProb b) { return a.prob < b.prob; };

struct path_cmp {
    bool operator()(const PathProb &a, const PathProb &b) const
    {
        return a.prob < b.prob;
    }
};

class CTCdecode {
  private:
    float *ctc_weight;
    float *ctc_bias;
    int vocab_size;

    set<PathProb, path_cmp> curr_hyps_set;
    deque<PathProb> hyps;
    void ctc_beam_search(Tensor<float> *din);
    int abs_time_step;

  public:
    CTCdecode(float *ctc_weight, float *ctc_bias, int vocab_size);
    ~CTCdecode();
    void forward(Tensor<float> *din);
    void show_hyps();
    void reset();
    vector<int> get_one_best_hyps();

    deque<PathProb> get_hyps();
};

#endif

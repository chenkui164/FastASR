
#ifndef PARAFORMER_MODELIMP_H
#define PARAFORMER_MODELIMP_H

#include <Model.h>
#include <stdint.h>
#include <string>

#include "../FeatureExtract.h"
#include "../Tensor.h"
#include "../Vocab.h"
#include "Decoder.h"
#include "Encoder.h"
#include "ModelParams.h"
#include "Predictor.h"

using namespace paraformer;

namespace paraformer {

class ModelImp : public Model {
  private:
    FeatureExtract *fe;
    paraformer::ModelParamsHelper *p_helper;
    Encoder *encoder;
    Predictor *predictor;
    Decoder *decoder;
    Vocab *vocab;

    void apply_lfr(Tensor<float> *&din);
    void apply_cmvn(Tensor<float> *din);

    string greedy_search(Tensor<float> *&encoder_out);

  public:
    ModelImp(const char *path, int mode);
    ~ModelImp();
    void reset();
    string forward_chunk(float *din, int len, int flag);
    string forward(float *din, int len, int flag);
    string rescoring();

};

} // namespace paraformer
#endif

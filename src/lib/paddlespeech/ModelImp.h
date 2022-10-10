
#ifndef PS_MODELIMP_H
#define PS_MODELIMP_H

#include <Model.h>
#include <stdint.h>
#include <string>

#include "../FeatureExtract.h"
#include "../Tensor.h"
#include "../Vocab.h"
#include "CTCDecode.h"
#include "Decoder.h"
#include "Encoder.h"
#include "ModelParams.h"

namespace paddlespeech {

class ModelImp : public Model {
  private:
    WenetParams params;
    FeatureExtract *fe;
    float *params_addr;
    int vocab_size;
    Encoder *encoder;
    Decoder *decoder;
    CTCdecode *ctc;
    PositionEncoding *pos_enc;
    Vocab *vocab;
    Tensor<float> *encoder_out_cache;

    void loadparams(const char *filename);
    void params_init();

    void hyps_process(deque<PathProb> hyps, Tensor<float> *din,
                      Tensor<int> *&hyps_pad, Tensor<int> *&hyps_mask,
                      Tensor<float> *&encoder_out, Tensor<int> *&encoder_mask);

    void calc_score(deque<PathProb> hyps, Tensor<float> *decoder_out,
                    Tensor<float> *scorce);

  public:
    ModelImp(const char *path, int mode);
    ~ModelImp();
    void reset();
    string forward_chunk(float *din, int len, int flag);
    string forward(float *din, int len, int flag);
    string rescoring();
};

} // namespace paddlespeech
#endif

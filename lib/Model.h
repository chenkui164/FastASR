
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <string>

#include "CTCDecode.h"
#include "Decoder.h"
#include "Encoder.h"
#include "Tensor.h"
#include "Vocab.h"
#include "WenetParams.h"

struct ModelConfig {
    const char *vocab_path;
    const char *wenet_path;
};

class Model {
  private:
    WenetParams params;
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
    Model(ModelConfig config, int mode);
    ~Model();
    void reset();
    string forward_chunk(Tensor<float> *din);
    string forward(Tensor<float> *din);
    string rescoring();
};

#endif

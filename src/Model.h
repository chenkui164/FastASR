
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

class Model {
  private:
    WenetParams *params;
    Encoder *encoder;
    Decoder *decoder;
    CTCdecode *ctc;
    PositionEncoding *pos_enc;
    Vocab *vocab;

    void loadparams(const char *filename);

  public:
    Model();
    ~Model();
    string forward(Tensor<float> *din, Tensor<float> *dout);
};

#endif

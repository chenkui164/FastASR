
#ifndef ENCODER_H
#define ENCODER_H

#include <stdint.h>

#include "EmbedLayer.h"
#include "PositionEncoding.h"
#include "SubEncoder.h"
#include "Tensor.h"
#include "WenetParams.h"

class Encoder {
  private:
    int cache_size;
    EncoderParams *params;
    EmbedLayer *embed;
    PositionEncoding *pos_enc;
    SubEncoder *subencoder[12];
    LayerNorm *after_norm;

  public:
    Encoder(EncoderParams *params, PositionEncoding *pos_enc, int mode);
    ~Encoder();
    void reset();
    void forward(Tensor<float> *&din);
};

#endif

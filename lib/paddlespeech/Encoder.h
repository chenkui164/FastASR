
#ifndef PS_ENCODER_H
#define PS_ENCODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "EmbedLayer.h"
#include "ModelParams.h"
#include "PositionEncoding.h"
#include "SubEncoder.h"

using namespace paddlespeech;

namespace paddlespeech {

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

} // namespace paddlespeech

#endif

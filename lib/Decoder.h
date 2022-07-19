
#ifndef DECODER_H
#define DECODER_H

#include <stdint.h>

#include "DecEmbedLayer.h"
#include "PositionEncoding.h"
#include "SubDecoder.h"
#include "SubEncoder.h"
#include "Tensor.h"
#include "WenetParams.h"

class Decoder {
  private:
    int vocab_size;
    DecoderParams *params;
    DecEmbedLayer *embed;
    PositionEncoding *pos_enc;
    SubDecoder *sub_decoder[6];
    LayerNorm *norm_after;

  public:
    Decoder(DecoderParams *params, PositionEncoding *pos_enc, int vocab_size);
    ~Decoder();
    void forward(Tensor<int> *hyps_pad, Tensor<int> *pad_len,
                 Tensor<float> *encoder_out, Tensor<int> *encoder_mask,
                 Tensor<float> *&dout);
};

#endif

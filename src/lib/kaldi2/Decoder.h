
#ifndef K2_DECODER_H
#define K2_DECODER_H

#include <stdint.h>

#include "../Tensor.h"
#include "ModelParams.h"

using namespace kaldi2;
namespace kaldi2 {

class Decoder {
  private:
    int vocab_size;
    DecoderParams *params;

  public:
    Decoder(DecoderParams *params, int vocab_size);
    ~Decoder();
    void forward(int *hyps, Tensor<float> *&dout);
};

} // namespace kaldi2

#endif

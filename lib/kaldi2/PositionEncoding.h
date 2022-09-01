

#ifndef PS_POSITIONENCODING_H
#define PS_POSITIONENCODING_H

#include "../Tensor.h"
#include <stdint.h>

namespace kaldi2 {
class PositionEncoding {
  private:
    Tensor<float> *pos_enc;

  public:
    PositionEncoding(int max);
    ~PositionEncoding();
    void fetch(int size, Tensor<float> *&out);
};
} // namespace kaldi2

#endif

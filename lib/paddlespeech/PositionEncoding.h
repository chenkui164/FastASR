

#ifndef PS_POSITIONENCODING_H
#define PS_POSITIONENCODING_H

#include "../Tensor.h"
#include <stdint.h>

class PositionEncoding {
  private:
    Tensor<float> *pos_enc;

  public:
    PositionEncoding(int max);
    ~PositionEncoding();
    void fetch(int size, Tensor<float> *&out);
};

#endif

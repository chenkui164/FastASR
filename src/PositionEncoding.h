

#ifndef POSITIONENCODING_H
#define POSITIONENCODING_H

#include <stdint.h>
#include "Tensor.h"

class PositionEncoding {
  private:
    Tensor<float> *pos_enc;

  public:
    PositionEncoding(int max);
    ~PositionEncoding();
    void fetch(int size, Tensor<float> *&out);

};

#endif

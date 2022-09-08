
#include <math.h>
#include <string.h>

#include "../Tensor.h"
#include "../predefine_coe.h"
#include "PositionEncoding.h"

using namespace kaldi2;

PositionEncoding::PositionEncoding(int max)
{
    pos_enc = new Tensor<float>(2 * max - 1, 512);
    float *div_term = (float *)pos_enc_div_term_hex;
    int i, j;
    int ii = 0;
    for (i = max - 1; i >= -max + 1; i--, ii++) {
        for (j = 0; j < 256; j++) {
            float coe = i * div_term[j];
            int idx = ii * 512 + 2 * j;
            pos_enc->buff[idx] = sin(coe);
            pos_enc->buff[idx + 1] = cos(coe);
        }
    }
}

PositionEncoding::~PositionEncoding()
{
    delete pos_enc;
}

void PositionEncoding::fetch(int size, Tensor<float> *&out)
{
    int all_size = size * 2 - 1;
    out = new Tensor<float>(all_size, 512);
    int start = pos_enc->size[2] / 2 - size + 1;

    memcpy(out->buff, pos_enc->buff + start * 512,
           all_size * 512 * sizeof(float));
}

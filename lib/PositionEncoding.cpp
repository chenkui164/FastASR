
#include <math.h>
#include <string.h>

#include "PositionEncoding.h"
#include "Tensor.h"
#include "predefine_coe.h"

PositionEncoding::PositionEncoding(int max)
{
    pos_enc = new Tensor<float>(max, 512);
    float *div = (float *)pos_enc_coe_hex;
    int i, j;
    for (i = 0; i < max; i++) {
        int offset = i * 512;
        float *buff = pos_enc->buff + offset;
        for (j = 0; j < 256; j++) {
            float tmp = i / div[j];
            buff[2 * j] = sin(tmp);
            buff[2 * j + 1] = cos(tmp);
        }
    }
}

PositionEncoding::~PositionEncoding()
{
    delete pos_enc;
}

void PositionEncoding::fetch(int size, Tensor<float> *&out)
{
    // out = new Tensor<float>(size, 512);
    // memcpy(out->buff, pos_enc->buff, out->buff_size * sizeof(float));
    out = pos_enc;
    out->resize(1, 1, size, 512);
}

#include "DecEmbedLayer.h"
#include <stdio.h>
#include <string.h>

DecEmbedLayer::DecEmbedLayer(float *params) : params(params)
{
}
DecEmbedLayer::~DecEmbedLayer()
{
}
void DecEmbedLayer::forward(Tensor<int> *din, Tensor<float> *&dout)
{
    dout = new Tensor<float>(din->size[2], din->size[3], 512);

    int mm = din->buff_size;
    int i;
    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        int offset2 = din->buff[i] * 512;
        memcpy(dout->buff + offset, params + offset2, sizeof(float) * 512);
    }
}

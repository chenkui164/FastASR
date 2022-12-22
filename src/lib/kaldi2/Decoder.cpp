#include "Decoder.h"
#include "../util.h"
#include <cblas.h>
#include <stdio.h>

using namespace kaldi2;

Decoder::Decoder(DecoderParams *params, int vocab_size)
    : params(params), vocab_size(vocab_size)
{
}

Decoder::~Decoder()
{
}

void Decoder::forward(int *hyps, Tensor<float> *&dout)
{
    Tensor<float> embed_out(2, 512);
    int i;
    for (i = 0; i < 2; i++) {
        int offset1 = i * 512;
        int offset2 = hyps[i] * 512;
        memcpy(embed_out.buff + offset1, params->embedding_weight + offset2,
               512 * sizeof(float));
    }


    for (i = 0; i < 512; i++) {
        int ii = i;
        int jj = i + 512;

        float val = embed_out.buff[ii] * params->conv_weight[ii] +
                    embed_out.buff[jj] * params->conv_weight[jj];
        dout->buff[i] = val < 0 ? 0 : val;
    }
}

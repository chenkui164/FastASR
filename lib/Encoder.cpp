#include <cblas.h>
#include <iostream>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include "Encoder.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "util.h"

using namespace std;

Encoder::Encoder(EncoderParams *params, PositionEncoding *pos_enc, int mode)
    : params(params), pos_enc(pos_enc)
{
    cache_size = 0;
    embed = new EmbedLayer(&params->embed);
    int i;
    for (i = 0; i < 12; i++) {
        subencoder[i] = new SubEncoder(&params->sub_encoder[i], mode);
    }
    after_norm = new LayerNorm(&params->after_norm, 1e-12f);
}

Encoder::~Encoder()
{
    delete embed;
}

void Encoder::reset()
{

    int i;
    cache_size = 0;
    for (i = 0; i < 12; i++) {
        subencoder[i]->reset();
    }
}

void Encoder::forward(Tensor<float> *&din)
{

    cache_size += din->size[2];

    embed->forward(din);
    Tensor<float> *pe_code;
    pos_enc->fetch(cache_size, pe_code);
    int i;
    for (i = 0; i < 12; i++) {
        subencoder[i]->forward(din, pe_code);
    }
    after_norm->forward(din);
}

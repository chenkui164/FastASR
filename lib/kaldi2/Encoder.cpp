#include <cblas.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "../Tensor.h"
#include "../util.h"
#include "Encoder.h"

using namespace std;
using namespace kaldi2;

Encoder::Encoder(EncoderParams *params, PositionEncoding *pos_enc, int mode)
    : params(params), pos_enc(pos_enc)
{
    embed = new EmbedLayer(&params->embed);
    int i;
    for (i = 0; i < 12; i++) {
        subencoder[i] = new SubEncoder(&params->sub_encoder[i], mode);
    }
}

Encoder::~Encoder()
{
    delete embed;
}

void Encoder::reset()
{
}

void Encoder::forward(Tensor<float> *&din)
{
    embed->forward(din);
    int Tmax = din->size[2];
    Tensor<float> *pe_code;
    pos_enc->fetch(Tmax, pe_code);
    int i;
    for (i = 0; i < 12; i++) {
        subencoder[i]->forward(din, pe_code);
    }
}


#include "SubEncoder.h"
#include "../util.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"

using namespace paraformer;

SubEncoder::SubEncoder(SubEncoderParams *params, int size) : params(params)
{
    norm1 = new LayerNorm(&params->norm1, 1e-12, size);
    self_attn = new EncSelfAttn(&params->self_attn);
    norm2 = new LayerNorm(&params->norm2, 1e-12, 512);
    feedforward = new FeedForward(&params->feedforward, 0);
}

SubEncoder::~SubEncoder()
{
}

void SubEncoder::reset()
{
}

void SubEncoder::forward(Tensor<float> *&din, int *conv_im2col)
{
    int mm = din->size[3];
    Tensor<float> residual(din);
    norm1->forward(din);
    Tensor<float> *mem;
    self_attn->forward(din, mem, conv_im2col);
    if (mm == 512) {
        din->add(mem, &residual);
    } else {
        din->add(mem);
    }
    delete mem;

    Tensor<float> residual2(din);
    norm2->forward(din);
    feedforward->forward(din);
    din->add(&residual2);
}

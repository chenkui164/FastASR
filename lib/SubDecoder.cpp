#include "SubDecoder.h"
#include "DecSelfAttn.h"
#include "util.h"

SubDecoder::SubDecoder(SubDecoderParams *params) : params(params)
{
    self_attn = new DecSelfAttn(&params->self_attn);
    src_attn = new DecSelfAttn(&params->src_attn);
    feedforward = new FeedForward(&params->feedward, 0);
    norm1 = new LayerNorm(&params->norm1, 1e-12);
    norm2 = new LayerNorm(&params->norm2, 1e-12);
    norm3 = new LayerNorm(&params->norm3, 1e-12);
}
SubDecoder::~SubDecoder()
{
}

void SubDecoder::forward(Tensor<float> *din, Tensor<int> *din_mask,
                         Tensor<float> *encoder_out, Tensor<int> *encoder_mask)
{
    Tensor<float> residual(din);
    norm1->forward(din);

    self_attn->forward(din, din, din, din_mask);

    din->add(1, &residual);
    residual.reload(din);
    norm2->forward(din);

    src_attn->forward(din, encoder_out, encoder_out, encoder_mask);
    // encoder_mask->disp();

    din->add(1, &residual);
    residual.reload(din);
    norm3->forward(din);
    feedforward->forward(din);
    din->add(1, &residual);
}

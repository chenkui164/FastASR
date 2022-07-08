
#include "SubEncoder.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"

SubEncoder::SubEncoder(SubEncoderParams *params) : params(params)
{

    norm_macaron = new LayerNorm(&params->norm_macaron, 1e-12f);
    norm_mha = new LayerNorm(&params->norm_mha, 1e-12f);
    norm_conv = new LayerNorm(&params->norm_conv, 1e-12f);
    norm_ff = new LayerNorm(&params->norm_ff, 1e-12f);
    norm_final = new LayerNorm(&params->norm_final, 1e-12f);
    feedforward_macron = new FeedForward(&params->feedforward_macaron, 1);
    feedforward = new FeedForward(&params->feedforward, 1);
    self_attn = new EncSelfAttn(&params->self_attn);
    conv_module = new ConvModule(&params->conv_module);
}

SubEncoder::~SubEncoder()
{
}

void SubEncoder::forward(Tensor<float> *din, Tensor<float> *pe)
{
    Tensor<float> residual(din);
    norm_macaron->forward(&residual);
    feedforward_macron->forward(&residual);
    din->add(0.5, &residual);
    residual.reload(din);
    norm_mha->forward(din);
    self_attn->forward(din, din, din, pe, din);
    din->add(1, &residual);
    residual.reload(din);
    norm_conv->forward(din);
    conv_module->forward(din);
    din->add(1, &residual);
    residual.reload(din);
    norm_ff->forward(&residual);
    feedforward->forward(&residual);
    din->add(0.5, &residual);
    norm_final->forward(din);
}

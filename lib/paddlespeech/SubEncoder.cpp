
#include "SubEncoder.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "LayerNorm.h"

using namespace paddlespeech;

SubEncoder::SubEncoder(SubEncoderParams *params, int mode) : params(params)
{
    in_cache = new Tensor<float>(1024, 512);
    in_cache->resize(1, 1, 0, 512);

    norm_macaron = new LayerNorm(&params->norm_macaron, 1e-12f);
    norm_mha = new LayerNorm(&params->norm_mha, 1e-12f);
    norm_conv = new LayerNorm(&params->norm_conv, 1e-12f);
    norm_ff = new LayerNorm(&params->norm_ff, 1e-12f);
    norm_final = new LayerNorm(&params->norm_final, 1e-12f);
    feedforward_macron = new FeedForward(&params->feedforward_macaron, 1);
    feedforward = new FeedForward(&params->feedforward, 1);
    self_attn = new EncSelfAttn(&params->self_attn);
    conv_module = new ConvModule(&params->conv_module, mode);
}

SubEncoder::~SubEncoder()
{
}

void SubEncoder::reset()
{
    in_cache->resize(1, 1, 0, 512);
    conv_module->reset();
}

void SubEncoder::forward(Tensor<float> *din, Tensor<float> *pe)
{

    Tensor<float> residual(din);

    norm_macaron->forward(&residual);
    feedforward_macron->forward(&residual);
    din->add(0.5, &residual);

    residual.reload(din);
    norm_mha->forward(din);
    int offset = in_cache->buff_size;
    in_cache->concat(din, 2);
    self_attn->forward(din, in_cache, in_cache, pe);
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

// void SubEncoder::forward(Tensor<float> *din, Tensor<float> *pe)
// {
//     int offset = in_cache->buff_size;
//     in_cache->concat(din, 2);

//     Tensor<float> residual(in_cache);
//     Tensor<float> tmp_in(in_cache);

//     norm_macaron->forward(&residual);
//     feedforward_macron->forward(&residual);
//     tmp_in.add(0.5, &residual);

//     residual.reload(&tmp_in);
//     norm_mha->forward(&tmp_in);

//     Tensor<float> x_q(din->size[2], din->size[3]);
//     memcpy(x_q.buff, tmp_in.buff + offset, x_q.buff_size * sizeof(float));

//     self_attn->forward(&x_q, &tmp_in, &tmp_in, pe, din);

//     // din->add(1, &residual);
//     // residual.reload(din);
//     // norm_conv->forward(din);
//     // conv_module->forward(din);
//     // din->add(1, &residual);
//     // residual.reload(din);
//     // norm_ff->forward(&residual);
//     // feedforward->forward(&residual);
//     // din->add(0.5, &residual);
//     // norm_final->forward(din);
// }

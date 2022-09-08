
#include "SubEncoder.h"
#include "../util.h"
#include "EncSelfAttn.h"
#include "FeedForward.h"

using namespace kaldi2;

SubEncoder::SubEncoder(SubEncoderParams *params, int mode) : params(params)
{
    in_cache = new Tensor<float>(1024, 512);
    in_cache->resize(1, 1, 0, 512);

    feedforward_macron = new FeedForward(&params->feedforward_macaron);
    feedforward = new FeedForward(&params->feedforward);
    self_attn = new EncSelfAttn(&params->self_attn);
    conv_module = new ConvModule(&params->conv_module, 0);
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

    feedforward_macron->forward(&residual);
    din->add(1, &residual);

    residual.reload(din);
    self_attn->forward(din, pe);
    din->add(1, &residual);

    residual.reload(din);
    conv_module->forward(din);
    din->add(1, &residual);

    residual.reload(din);
    feedforward->forward(&residual);
    din->add(1, &residual);

    basic_norm(din, *params->norm);

}

#include "ModelParams.h"
#include "../util.h"

using namespace kaldi2;

ModelParamsHelper::ModelParamsHelper(const char *path, int vocab_size)
    : vocab_size(vocab_size)
{
    params_addr = loadparams(path);
    offset = 0;
    params_init(params);
}

ModelParamsHelper::~ModelParamsHelper()
{
}

float *ModelParamsHelper::get_addr(int num)
{
    float *tmp = params_addr + offset;
    offset += val_align(num, 32);
    return tmp;
}

void ModelParamsHelper::params_init(ModelParams &p_in)
{
    param_init_encoder(p_in.encoder);
}

void ModelParamsHelper::param_init_encoder(EncoderParams &p_in)
{
    param_init_encoder_embed(p_in.embed);
}

void ModelParamsHelper::param_init_encoder_embed(EncEmbedParams &p_in)
{
    p_in.conv0_weight = get_addr(72);
    p_in.conv0_bias = get_addr(8);

    p_in.conv1_weight = get_addr(2304);
    p_in.conv1_bias = get_addr(32);

    p_in.conv2_weight = get_addr(36864);
    p_in.conv2_bias = get_addr(128);

    p_in.out_weight = get_addr(512 * 2432);
    p_in.out_bias = get_addr(512);

    p_in.out_norm = get_addr(1);
}

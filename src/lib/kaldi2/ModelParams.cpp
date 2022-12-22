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
    aligned_free(params_addr);
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
    param_init_decoder(p_in.decoder);
    param_init_joiner(p_in.joiner);
}

void ModelParamsHelper::param_init_encoder(EncoderParams &p_in)
{
    param_init_encoder_embed(p_in.embed);
    int i;
    for (i = 0; i < 12; i++) {
        param_init_encoder_subencoder(p_in.sub_encoder[i]);
    }
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

void ModelParamsHelper::param_init_encoder_subencoder(SubEncoderParams &p_in)
{

    param_init_encoder_selfattn(p_in.self_attn);
    param_init_feedforward(p_in.feedforward);
    param_init_feedforward(p_in.feedforward_macaron);
    param_init_encoder_conv(p_in.conv_module);
    p_in.norm = get_addr(1);
}

void ModelParamsHelper::param_init_encoder_selfattn(EncSelfAttnParams &p_in)
{
    p_in.pos_bias_u = get_addr(8 * 64);
    p_in.pos_bias_v = get_addr(8 * 64);

    p_in.in_proj_weight = get_addr(1536 * 512);
    p_in.in_proj_bias = get_addr(1536);

    p_in.out_proj_weight = get_addr(512 * 512);
    p_in.out_proj_bias = get_addr(512);

    p_in.linear_pos_weight = get_addr(512 * 512);
}

void ModelParamsHelper::param_init_feedforward(FeedForwardParams &p_in)
{
    p_in.w1_weight = get_addr(2048 * 512);
    p_in.w1_bias = get_addr(2048);

    p_in.w2_weight = get_addr(512 * 2048);
    p_in.w2_bias = get_addr(512);
}

void ModelParamsHelper::param_init_encoder_conv(EncConvParams &p_in)
{
    p_in.pointwise_conv1_weight = get_addr(1024 * 512);
    p_in.pointwise_conv1_bias = get_addr(1024);

    p_in.depthwise_conv_weight = get_addr(512 * 31);
    p_in.depthwise_conv_bias = get_addr(512);

    p_in.pointwise_conv2_weight = get_addr(512 * 512);
    p_in.pointwise_conv2_bias = get_addr(512);
}

void ModelParamsHelper::param_init_decoder(DecoderParams &p_in)
{
    p_in.embedding_weight = get_addr(5537 * 512);
    p_in.conv_weight = get_addr(512 * 2);
}

void ModelParamsHelper::param_init_joiner(JoinerParams &p_in)
{

    p_in.encoder_proj_weight = get_addr(512 * 512);
    p_in.encoder_proj_bias = get_addr(512);
    p_in.decoder_proj_weight = get_addr(512 * 512);
    p_in.decoder_proj_bias = get_addr(512);
    p_in.output_linear_weight = get_addr(5537 * 512);
    p_in.output_linear_bias = get_addr(5537);
}

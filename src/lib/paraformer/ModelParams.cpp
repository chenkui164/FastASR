#include "ModelParams.h"
#include "../util.h"

using namespace paraformer;

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
    param_init_predictor(p_in.predictor);
    param_init_decoder(p_in.decoder);
}

void ModelParamsHelper::param_init_encoder(EncoderParams &p_in)
{
    param_init_encoder_subencoder(p_in.sub_encoders0, 560);
    int i;
    for (i = 0; i < 49; i++) {
        param_init_encoder_subencoder(p_in.sub_encoders[i], 512);
    }
    param_init_layernorm(p_in.after_norm, 512);
}

void ModelParamsHelper::param_init_encoder_subencoder(SubEncoderParams &p_in,
                                                      int size)
{

    param_init_feedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, size);
    param_init_layernorm(p_in.norm2, 512);

    param_init_encoder_selfattn(p_in.self_attn, size);
}

void ModelParamsHelper::param_init_encoder_selfattn(EncSelfAttnParams &p_in,
                                                    int size)
{
    p_in.fsmn_block_weight = get_addr(512 * 11);
    p_in.linear_out_bias = get_addr(512);
    p_in.linear_out_weight = get_addr(512 * 512);
    p_in.linear_qkv_bias = get_addr(1536);
    p_in.linear_qkv_weight = get_addr(1536 * size);
}

void ModelParamsHelper::param_init_feedforward(FeedForwardParams &p_in)
{
    p_in.w1_bias = get_addr(2048);
    p_in.w1_weight = get_addr(2048 * 512);

    p_in.w2_bias = get_addr(512);
    p_in.w2_weight = get_addr(512 * 2048);
}

void ModelParamsHelper::param_init_decoderfeedforward(
    DecoderFeedForwardParams &p_in)
{
    param_init_layernorm(p_in.norm, 2048);
    p_in.w1_bias = get_addr(2048);
    p_in.w1_weight = get_addr(2048 * 512);
    p_in.w2_weight = get_addr(512 * 2048);

}

void ModelParamsHelper::param_init_layernorm(NormParams &p_in, int size)
{
    p_in.bias = get_addr(size);
    p_in.weight = get_addr(size);
}

void ModelParamsHelper::param_init_decoder(DecoderParams &p_in)
{
    int i;
    for (i = 0; i < 16; i++) {
        param_init_subdecoder(p_in.sub_decoders[i]);
    }

    param_init_subdecoder3(p_in.sub_decoders3);
    param_init_layernorm(p_in.after_norm, 512);

    p_in.linear_out_bias = get_addr(8404);
    p_in.linear_out_weight = get_addr(8404 * 512);
}

void ModelParamsHelper::param_init_subdecoder(SubDecoderParams &p_in)
{
    param_init_decoderfeedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, 512);
    param_init_layernorm(p_in.norm2, 512);
    param_init_layernorm(p_in.norm3, 512);
    p_in.fsmn_block_weight = get_addr(512 * 11);
    param_init_decselfattn(p_in.src_attn);
}

void ModelParamsHelper::param_init_subdecoder3(SubDecoder3Params &p_in)
{
    param_init_decoderfeedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, 512);
}

void ModelParamsHelper::param_init_decselfattn(DecSelfAttnParams &p_in)
{
    p_in.linear_kv_bias = get_addr(1024);
    p_in.linear_kv_weight = get_addr(1024 * 512);

    p_in.linear_out_bias = get_addr(512);
    p_in.linear_out_weight = get_addr(512 * 512);

    p_in.linear_q_bias = get_addr(512);
    p_in.linear_q_weight = get_addr(512 * 512);
}

void ModelParamsHelper::param_init_predictor(PredictorParams &p_in)
{
    p_in.cif_conv1d_bias = get_addr(512);
    p_in.cif_conv1d_weight = get_addr(512 * 512 * 3);

    p_in.cif_output_bias = get_addr(1);
    p_in.cif_output_weight = get_addr(512);
}

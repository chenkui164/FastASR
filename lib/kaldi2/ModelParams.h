
#ifndef K2_MODELPARAMS_H
#define K2_MODELPARAMS_H

#include <stdint.h>
namespace kaldi2 {

typedef struct {
    float *conv0_weight;
    float *conv0_bias;

    float *conv1_weight;
    float *conv1_bias;

    float *conv2_weight;
    float *conv2_bias;

    float *out_weight;
    float *out_bias;

    float *out_norm;

} EncEmbedParams;

typedef struct {
    float *pos_bias_u;
    float *pos_bias_v;
    float *in_proj_weight;
    float *in_proj_bias;
    float *out_proj_weight;
    float *out_proj_bias;
    float *linear_pos_weight;
} EncSelfAttnParams;

typedef struct {
    float *w1_weight;
    float *w1_bias;
    float *w2_weight;
    float *w2_bias;
} FeedForwardParams;

typedef struct {
    float *pointwise_conv1_weight;
    float *pointwise_conv1_bias;

    float *depthwise_conv_weight;
    float *depthwise_conv_bias;

    float *pointwise_conv2_weight;
    float *pointwise_conv2_bias;
} EncConvParams;

typedef struct {
    EncSelfAttnParams self_attn;
    FeedForwardParams feedforward;
    FeedForwardParams feedforward_macaron;
    EncConvParams conv_module;
    float *norm;
} SubEncoderParams;

typedef struct {
    EncEmbedParams embed;
    SubEncoderParams sub_encoder[12];
} EncoderParams;

typedef struct {
    float *embedding_weight;
    float *conv_weight;
} DecoderParams;

typedef struct {
    float *encoder_proj_weight;
    float *encoder_proj_bias;
    float *decoder_proj_weight;
    float *decoder_proj_bias;
    float *output_linear_weight;
    float *output_linear_bias;
} JoinerParams;

typedef struct {
    EncoderParams encoder;
    DecoderParams decoder;
    JoinerParams joiner;
} ModelParams;

class ModelParamsHelper {
  private:
    float *params_addr;
    int offset;
    int vocab_size;

    float *get_addr(int num);

  public:
    ModelParamsHelper(const char *path, int vocab_size);
    ~ModelParamsHelper();
    ModelParams params;
    void params_init(ModelParams &p_in);
    void param_init_encoder(EncoderParams &p_in);
    void param_init_encoder_embed(EncEmbedParams &p_in);
    void param_init_encoder_subencoder(SubEncoderParams &p_in);
    void param_init_encoder_selfattn(EncSelfAttnParams &p_in);
    void param_init_encoder_conv(EncConvParams &p_in);
    void param_init_feedforward(FeedForwardParams &p_in);
    void param_init_decoder(DecoderParams &p_in);
    void param_init_joiner(JoinerParams &p_in);
};

} // namespace kaldi2
#endif

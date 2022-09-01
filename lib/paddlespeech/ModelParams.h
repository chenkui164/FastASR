
#ifndef PS_MODELPARAMS_H
#define PS_MODELPARAMS_H

namespace paddlespeech {

typedef struct {
    float *conv0_weight;
    float *conv0_bias;

    float *conv1_weight;
    float *conv1_bias;

    float *out0_weight;
    float *out0_bias;

} EncEmbedParams;

typedef struct {
    float *linear_q_weight;
    float *linear_q_bias;
    float *linear_k_weight;
    float *linear_k_bias;
    float *linear_v_weight;
    float *linear_v_bias;
    float *linear_out_weight;
    float *linear_out_bias;
} SelfAttnParams;

typedef struct {
    SelfAttnParams linear0;
    float *linear_pos_weight;
    float *pos_bias_u;
    float *pos_bias_v;
} EncSelfAttnParams;

typedef struct {
    float *w1_weight;
    float *w1_bias;
    float *w2_weight;
    float *w2_bias;
} FeedForwardParams;

typedef struct {
    float *weight;
    float *bias;
} NormParams;

typedef struct {
    float *pointwise_conv1_weight;
    float *pointwise_conv1_bias;

    float *depthwise_conv_weight;
    float *depthwise_conv_bias;

    float *pointwise_conv2_weight;
    float *pointwise_conv2_bias;
    NormParams norm;
} EncConvParams;

typedef struct {
    EncSelfAttnParams self_attn;
    FeedForwardParams feedforward;
    FeedForwardParams feedforward_macaron;
    EncConvParams conv_module;
    NormParams norm_ff;
    NormParams norm_mha;
    NormParams norm_macaron;
    NormParams norm_conv;
    NormParams norm_final;
    // float concat_weight[1024 * 512];
    // float concat_bias[512];
} SubEncoderParams;

typedef struct {
    EncEmbedParams embed;
    SubEncoderParams sub_encoder[12];
    NormParams after_norm;
} EncoderParams;

typedef struct {
    SelfAttnParams self_attn;
    SelfAttnParams src_attn;
    FeedForwardParams feedward;
    NormParams norm1;
    NormParams norm2;
    NormParams norm3;
    // float concat_weight1[1024 * 512];
    // float concat_bias1[512];
    // float concat_weight2[1024 * 512];
    // float concat_bias2[512];
} SubDecoderParams;

typedef struct {
    float *embed_weight;
    SubDecoderParams sub_decoder[6];
    NormParams after_norm;
    float *output_weight;
    float *output_bias;
} DecoderParams;

typedef struct {
    EncoderParams encoder;
    float *ctc_weight;
    float *ctc_bias;
    DecoderParams decoder;
} WenetParams;

} // namespace paddlespeech

#endif

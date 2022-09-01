
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
    EncEmbedParams embed;
} EncoderParams;

typedef struct {
    EncoderParams encoder;
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
};

} // namespace kaldi2
#endif

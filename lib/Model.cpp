#include <deque>
#include <fftw3.h>
#include <iostream>
#include <locale.h>
#include <malloc.h>
#include <math.h>
#include <string.h>

#include "CTCDecode.h"
#include "EmbedLayer.h"
#include "Encoder.h"
#include "Model.h"
#include "Tensor.h"
#include "Vocab.h"
#include "WenetParams.h"
#include "predefine_coe.h"
#include "util.h"

using namespace std;

void disp_params(float *din, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%f ", din[i]);
    }
    printf("\n");
}

Model::Model(ModelConfig cfg, int mode)
{
    fe = new FeatureExtract(mode);

    loadparams(cfg.wenet_path);
    vocab = new Vocab(cfg.vocab_path);
    vocab_size = vocab->size();
    params_init();

    pos_enc = new PositionEncoding(5000);
    encoder = new Encoder(&params.encoder, pos_enc, mode);

    ctc = new CTCdecode(params.ctc_weight, params.ctc_bias, vocab_size);
    decoder = new Decoder(&params.decoder, pos_enc, vocab_size);

    encoder_out_cache = new Tensor<float>(1024, 512);
    encoder_out_cache->resize(1, 1, 0, 512);

    // disp_params(params->decoder.sub_decoder[0].self_attn.linear_q_bias, 10);
}

Model::~Model()
{
    delete encoder;
    delete ctc;
    delete fe;
}

void Model::reset()
{
    encoder_out_cache->resize(1, 1, 0, 512);
    encoder->reset();
    ctc->reset();
    fe->reset();
}

void Model::hyps_process(deque<PathProb> hyps, Tensor<float> *din,
                         Tensor<int> *&hyps_pad, Tensor<int> *&hyps_mask,
                         Tensor<float> *&encoder_out,
                         Tensor<int> *&encoder_mask)
{
    int mm = hyps.size();

    int i = 0;
    int j = 0;
    int max = 0;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int ll = hyps_it->prefix.size() + 1;
        max = max > ll ? max : ll;
        i++;
    }

    hyps_pad = new Tensor<int>(mm, max);
    hyps_mask = new Tensor<int>(mm, max);

    encoder_out = new Tensor<float>(mm, din->size[2], din->size[3]);

    i = 0;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int ll = hyps_it->prefix.size() + 1;
        hyps_pad->buff[i * max] = vocab_size - 1;
        hyps_mask->buff[i * max] = 1;
        for (j = 1; j < max; j++) {
            int ii = i * max + j;
            hyps_pad->buff[ii] =
                j < ll ? hyps_it->prefix[j - 1] : vocab_size - 1;
            hyps_mask->buff[ii] = j < ll ? j + 1 : ll;
        }

        int offset = i * din->buff_size;
        memcpy(encoder_out->buff + offset, din->buff,
               sizeof(float) * din->buff_size);
        i++;
    }

    encoder_mask = new Tensor<int>(mm, max);
    for (i = 0; i < encoder_mask->buff_size; i++) {
        encoder_mask->buff[i] = din->size[2];
    }
}

void Model::calc_score(deque<PathProb> hyps, Tensor<float> *decoder_out,
                       Tensor<float> *scorce)
{

    int i = 0;
    scorce->zeros();
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int j;
        int ll = hyps_it->prefix.size();
        float *prob;

        for (j = 0; j < ll; j++) {
            int char_idx = hyps_it->prefix[j];
            prob = decoder_out->buff +
                   (i * decoder_out->buff_size / 10 + j * vocab_size);
            scorce->buff[i] = scorce->buff[i] + prob[char_idx];
        }

        prob = decoder_out->buff +
               (i * decoder_out->buff_size / 10 + j * vocab_size);
        scorce->buff[i] = scorce->buff[i] + prob[vocab_size - 1];
        i++;
    }
}

string Model::forward(short *din, int len, int flag)
{
    Tensor<float> *in;
    fe->insert(din, len, flag);
    fe->fetch(in);
    encoder->forward(in);
    encoder_out_cache->concat(in, 2);
    ctc->forward(in);

    return rescoring();
}

string Model::forward_chunk(short *din, int len, int flag)
{

    fe->insert(din, len, flag);
    if (fe->size() < 1) {
        vector<int> result = ctc->get_one_best_hyps();
        return vocab->vector2string(result);
    }

    Tensor<float> *in;
    fe->fetch(in);

    encoder->forward(in);
    encoder_out_cache->concat(in, 2);
    ctc->forward(in);
    delete in;
    vector<int> result = ctc->get_one_best_hyps();

    return vocab->vector2string(result);
}

string Model::rescoring()
{
    deque<PathProb> hyps = ctc->get_hyps();

    Tensor<int> *hyps_pad;
    Tensor<int> *hyps_mask;
    Tensor<float> *encoder_out;
    Tensor<int> *encoder_mask;

    hyps_process(hyps, encoder_out_cache, hyps_pad, hyps_mask, encoder_out,
                 encoder_mask);

    Tensor<float> *decoder_out;
    decoder->forward(hyps_pad, hyps_mask, encoder_out, encoder_mask,
                     decoder_out);
    delete hyps_pad;
    delete hyps_mask;
    delete encoder_out;
    delete encoder_mask;

    // decoder_out->dump();
    // hyps_pad->dump();

    Tensor<float> scorce(1, 10);
    scorce.zeros();
    calc_score(hyps, decoder_out, &scorce);
    delete decoder_out;

    int i = 0;
    float max = -INFINITY;
    vector<int> result;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        float tmp_scorce = 0.5 * hyps_it->prob + scorce.buff[i];
        printf("score is %f %f %f\n", tmp_scorce, hyps_it->prob,
               scorce.buff[i]);
        if (tmp_scorce > max) {
            max = tmp_scorce;
            result = hyps_it->prefix;
        }
        scorce.buff[i] = tmp_scorce;
        i++;
    }

    return vocab->vector2string(result);
}

// string Model::forward(Tensor<float> *din, Tensor<float> *dout)
// {
//     encoder->forward(din, dout);
//     deque<PathProb> hyps;
//     ctc->forward(dout, hyps);
//     // ctc->show_hyps(hyps);
//     Tensor<int> *hyps_pad;
//     Tensor<int> *hyps_mask;
//     Tensor<float> *encoder_out;
//     Tensor<int> *encoder_mask;
//     hyps_process(hyps, dout, hyps_pad, hyps_mask, encoder_out, encoder_mask);
//     // printf("hyps size is %ld\n", hyps.size());
//     Tensor<float> *decoder_out;
//     decoder->forward(hyps_pad, hyps_mask, encoder_out, encoder_mask,
//                      decoder_out);

//     Tensor<float> scorce(1, 10);
//     rescoring(hyps, decoder_out, &scorce);

//     int i = 0;
//     float max = -INFINITY;
//     vector<int> result;
//     for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
//         float tmp_scorce = 0.5 * hyps_it->prob + scorce.buff[i];
//         if (tmp_scorce > max) {
//             max = tmp_scorce;
//             result = hyps_it->prefix;
//         }
//         scorce.buff[i] = tmp_scorce;
//         i++;
//     }
//     // cout << vocab->vector2string(result) << endl;
//     return vocab->vector2string(result);
// }

void Model::loadparams(const char *filename)
{

    FILE *fp;
    fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    uint32_t nFileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    params_addr = (float *)memalign(32, nFileLen);
    int n = fread(params_addr, 1, nFileLen, fp);

    fclose(fp);
}

void param_init_layernorm(NormParams &p_in, float *base_addr, int &offset)
{

    p_in.weight = base_addr + offset;
    offset += 512;

    p_in.bias = base_addr + offset;
    offset += 512;
}

void param_init_feedforward(FeedForwardParams &p_in, float *base_addr,
                            int &offset)
{

    p_in.w1_weight = base_addr + offset;
    offset += 512 * 2048;

    p_in.w1_bias = base_addr + offset;
    offset += 2048;

    p_in.w2_weight = base_addr + offset;
    offset += 2048 * 512;

    p_in.w2_bias = base_addr + offset;
    offset += 512;
}

void param_init_encoder_embed(EncEmbedParams &p_in, float *base_addr,
                              int &offset)
{
    p_in.conv0_weight = base_addr + offset;
    offset += 512 * 9;

    p_in.conv0_bias = base_addr + offset;
    offset += 512;

    p_in.conv1_weight = base_addr + offset;
    offset += 512 * 512 * 9;

    p_in.conv1_bias = base_addr + offset;
    offset += 512;

    p_in.out0_weight = base_addr + offset;
    offset += 9728 * 512;

    p_in.out0_bias = base_addr + offset;
    offset += 512;
}

void param_init_self_attn(SelfAttnParams &p_in, float *base_addr, int &offset)
{
    p_in.linear_q_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.linear_q_bias = base_addr + offset;
    offset += 512;

    p_in.linear_k_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.linear_k_bias = base_addr + offset;
    offset += 512;

    p_in.linear_v_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.linear_v_bias = base_addr + offset;
    offset += 512;

    p_in.linear_out_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.linear_out_bias = base_addr + offset;
    offset += 512;
}

void param_init_enc_self_attn(EncSelfAttnParams &p_in, float *base_addr,
                              int &offset)
{
    param_init_self_attn(p_in.linear0, base_addr, offset);

    p_in.linear_pos_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.pos_bias_u = base_addr + offset;
    offset += 512;

    p_in.pos_bias_v = base_addr + offset;
    offset += 512;
}

void param_init_enc_conv_module(EncConvParams &p_in, float *base_addr,
                                int &offset)
{
    p_in.pointwise_conv1_weight = base_addr + offset;
    offset += 1024 * 512;

    p_in.pointwise_conv1_bias = base_addr + offset;
    offset += 1024;

    p_in.depthwise_conv_weight = base_addr + offset;
    offset += 512 * 15;

    p_in.depthwise_conv_bias = base_addr + offset;
    offset += 512;

    p_in.pointwise_conv2_weight = base_addr + offset;
    offset += 512 * 512;

    p_in.pointwise_conv2_bias = base_addr + offset;
    offset += 512;

    param_init_layernorm(p_in.norm, base_addr, offset);
}

void param_init_subencoder(SubEncoderParams &p_in, float *base_addr,
                           int &offset)
{
    param_init_enc_self_attn(p_in.self_attn, base_addr, offset);

    param_init_feedforward(p_in.feedforward, base_addr, offset);
    param_init_feedforward(p_in.feedforward_macaron, base_addr, offset);
    param_init_enc_conv_module(p_in.conv_module, base_addr, offset);

    param_init_layernorm(p_in.norm_ff, base_addr, offset);
    param_init_layernorm(p_in.norm_mha, base_addr, offset);
    param_init_layernorm(p_in.norm_macaron, base_addr, offset);
    param_init_layernorm(p_in.norm_conv, base_addr, offset);
    param_init_layernorm(p_in.norm_final, base_addr, offset);
}

void param_init_encoder(EncoderParams &p_in, float *base_addr, int &offset)
{
    param_init_encoder_embed(p_in.embed, base_addr, offset);
    int i;
    for (i = 0; i < 12; i++) {
        param_init_subencoder(p_in.sub_encoder[i], base_addr, offset);
    }
    param_init_layernorm(p_in.after_norm, base_addr, offset);
}

void param_init_subdecoder(SubDecoderParams &p_in, float *base_addr,
                           int &offset)
{
    param_init_self_attn(p_in.self_attn, base_addr, offset);
    param_init_self_attn(p_in.src_attn, base_addr, offset);
    param_init_feedforward(p_in.feedward, base_addr, offset);
    param_init_layernorm(p_in.norm1, base_addr, offset);
    param_init_layernorm(p_in.norm2, base_addr, offset);
    param_init_layernorm(p_in.norm3, base_addr, offset);
}

void param_init_decoder(DecoderParams &p_in, float *base_addr, int &offset,
                        int vocab_size)
{
    p_in.embed_weight = base_addr + offset;
    offset += 512 * vocab_size;

    int i;
    for (i = 0; i < 6; i++) {
        param_init_subdecoder(p_in.sub_decoder[i], base_addr, offset);
    }

    param_init_layernorm(p_in.after_norm, base_addr, offset);

    p_in.output_weight = base_addr + offset;
    offset += 512 * vocab_size;

    p_in.output_bias = base_addr + offset;
    offset += vocab_size;
}

void Model::params_init()
{
    int offset = 0;
    param_init_encoder(params.encoder, params_addr, offset);

    params.ctc_weight = params_addr + offset;
    offset += 512 * vocab_size;

    params.ctc_bias = params_addr + offset;
    offset += vocab_size;

    param_init_decoder(params.decoder, params_addr, offset, vocab_size);
}

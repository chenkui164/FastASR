#include "Decoder.h"
#include "util.h"
#include <cblas.h>
#include <stdio.h>

Decoder::Decoder(DecoderParams *params, PositionEncoding *pos_enc)
    : params(params), pos_enc(pos_enc)
{
    embed = new DecEmbedLayer(params->embed_weight);
    int i;
    for (i = 0; i < 6; i++) {
        sub_decoder[i] = new SubDecoder(&params->sub_decoder[i]);
    }
    norm_after = new LayerNorm(&params->after_norm, 1e-12f);
}

Decoder::~Decoder()
{
}

void Decoder::forward(Tensor<int> *&hyps_pad, Tensor<int> *&hyps_mask,
                      Tensor<float> *&encoder_out, Tensor<int> *&encoder_mask,
                      Tensor<float> *&dout)
{
    // printf("Decoder!!!!\n");

    int tmp = 0x41b504f3;
    float scale = *((float *)&tmp);

    Tensor<float> *embed_out;
    Tensor<float> *pos_code;
    embed->forward(hyps_pad, embed_out);
    pos_enc->fetch(embed_out->size[2], pos_code);
    int mm = embed_out->size[1];
    int i, j;
    for (i = 0; i < mm; i++) {
        int offset = pos_code->buff_size * i;
        float *buff = embed_out->buff + offset;
        for (j = 0; j < pos_code->buff_size; j++) {
            buff[j] = scale * buff[j] + pos_code->buff[j];
        }
    }

    for (i = 0; i < 6; i++) {
        sub_decoder[i]->forward(embed_out, hyps_mask, encoder_out,
                                encoder_mask);
    }


    norm_after->forward(embed_out);
    dout = new Tensor<float>(embed_out->size[1], embed_out->size[2], 5537);

    mm = dout->buff_size / 5537;
    for (i = 0; i < mm; i++) {
        int offset = i * 5537;
        memcpy(dout->buff + offset, params->output_bias, 5537 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 5537, 512, 1,
                embed_out->buff, 512, params->output_weight, 5537, 1,
                dout->buff, 5537);

    for (i = 0; i < mm; i++) {
        int offset = i * 5537;
        log_softmax(dout->buff + offset, 5537);
    }
}

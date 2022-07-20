#include "fastasr.h"
#include <malloc.h>
#include <stdio.h>

int main()
{

    int16_t *buff;
    int len;
    void *audio_obj = audio_create(0);
    audio_loadwav(audio_obj, "zh.wav");
    audio_disp(audio_obj);
    int flag = audio_fetch(audio_obj, &buff, &len);

    struct ModelConfig cfg;
    cfg.vocab_path = "./cli/vocab.txt";
    cfg.wenet_path = "./cli/wenet_params.bin";
    void *modle_obj = model_create(cfg, 0);
    model_reset(modle_obj);

    char *result = model_forward(modle_obj, buff, len, flag);
    printf("%s\n", result);

    audio_free(audio_obj);
    model_free(modle_obj);
    free(result);

    return 0;
}
